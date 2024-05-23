// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

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
    builder.SetMaxReceiveMessageSize(268435456);
    builder.SetMaxSendMessageSize(536870912);
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
