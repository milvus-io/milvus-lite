#include "milvus_service_impl.h"
#include <iostream>
#include "log/Log.h"
#include "string_util.hpp"
#include <string>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <fcntl.h>
#include <unistd.h>

int
BlockLock(const char* filename) {
    int fd = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        LOG_ERROR("Open lock file {} failed", filename);
        return -1;
    }
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    // block lock
    if (fcntl(fd, F_SETLKW, &fl) == -1) {
        close(fd);
        return -1;
    }
    // unlock file
    fl.l_type = F_UNLCK;
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        return -1;
    }
    LOG_ERROR("Process exit");
    close(fd);
    return 0;
}

int
main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (!(argc == 3 || argc == 4 || argc == 5)) {
        return -1;
    }

    std::string work_dir = argv[1];
    std::string address = argv[2];
    std::string log_level = "ERROR";
    if (argc == 4) {
        log_level = argv[3];
    }
    if (log_level == "INFO") {
        google::SetStderrLogging(google::INFO);
    } else {
        google::SetStderrLogging(google::ERROR);
    }

    ::milvus::local::MilvusServiceImpl service(work_dir);
    if (!service.Init()) {
        LOG_ERROR("Init milvus failed");
        return -1;
    }
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
    LOG_INFO("Start milvus-local success...");
    if (argc == 5) {
        auto filename = argv[4];
        /*
          Blocked while attempting to acquire a file lock held by the parent process.
          When the lock is successfully acquired, it indicates that the parent process has exited,
          and the child process should exit as well.
        */
        BlockLock(filename);
    } else {
        server->Wait();
    }
    return 0;
}
