#pragma once

#include <string>
#include "common/type_c.h"
#include "string_util.hpp"

namespace milvus::local {

// errors are the subset of milvus/pkg/merr/errors.go
enum ErrCode {
    Succ = 0,
    ErrServiceInternal = 5,
    ErrCollectionNotFound = 100,
    ErrCollectionNotLoaded = 101,
    ErrCollectionNumLimitExceeded = 102,
    ErrCollectionNotFullyLoaded = 103,
    ErrCollectionLoaded = 104,
    ErrCollectionIllegalSchema = 105,

    // not in milvus
    ErrCollectionAlreadExist = 199,

    ErrIndexNotFound = 700,
    ErrIndexNotSupported = 701,
    ErrIndexDuplicate = 702,

    ErrParameterInvalid = 1100,
    ErrParameterMissing = 1101,

    ErrMetricNotFound = 1200,

    ErrFieldNotFound = 1700,
    ErrFieldInvalidName = 1701,

    ErrMissingRequiredParameters = 1802,
    ErrMarshalCollectionSchema = 1803,
    ErrInvalidInsertData = 1804,
    ErrInvalidSearchResult = 1805,
    ErrCheckPrimaryKey = 1806,

    ErrSegcore = 2000,

    ErrUndefined = 65535

};

class Status {
 public:
    explicit Status(CStatus cstatus) {
        if (cstatus.error_code != 0) {
            msg_ = "segcore error";
            code_ = ErrSegcore;
            detail_ = cstatus.error_msg;
            free((void*)cstatus.error_msg);
            cstatus.error_msg = NULL;
        } else {
            code_ = 0;
            msg_ = "";
            detail_ = "";
        }
    }

 private:
    Status(int code, const std::string& msg, const std::string& detail = "")
        : code_(code), msg_(msg), detail_(detail) {
    }

 public:
    virtual ~Status() = default;

    Status(const Status& rhs) = delete;
    Status&
    operator=(const Status& rhs) = delete;

    Status(Status&& rhs) : code_(rhs.code_) {
        msg_ = std::move(rhs.msg_);
        detail_ = std::move(rhs.detail_);
    }

    Status&
    operator=(Status&& rhs);

 public:
    template <class... Args>
    static Status
    ServiceInternal(const std::string& detail = "", Args&&... args) {
        return Status(ErrServiceInternal,
                      "internal error",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    SegcoreErr(const std::string& detail = "", Args&&... args) {
        return Status(
            ErrSegcore, "segcore error", string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    CollectionNotFound(const std::string& detail = "", Args&&... args) {
        return Status(ErrCollectionNotFound,
                      "collection not found",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    CollectionAlreadExist(const std::string& detail = "", Args&&... args) {
        return Status(ErrCollectionAlreadExist,
                      "collection alread exists",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    CollectionNotLoaded(const std::string& detail = "", Args&&... args) {
        return Status(ErrCollectionNotLoaded,
                      "collection not loaded",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    CollectionLoaded(const std::string& detail = "", Args&&... args) {
        return Status(ErrCollectionLoaded,
                      "collection already loaded",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    CollectionIllegalSchema(const std::string& detail = "", Args&&... args) {
        return Status(ErrCollectionIllegalSchema,
                      "illegal collection schema",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    IndexNotFound(const std::string& detail = "", Args&&... args) {
        return Status(ErrIndexNotFound,
                      "index not found",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    IndexNotSupported(const std::string& detail = "", Args&&... args) {
        return Status(ErrIndexNotSupported,
                      "index type not supported",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    IndexDuplicate(const std::string& detail = "", Args&&... args) {
        return Status(ErrIndexDuplicate,
                      "index duplicates",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    ParameterInvalid(const std::string& detail = "", Args&&... args) {
        return Status(ErrParameterInvalid,
                      "invalid parameter",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    ParameterMissing(const std::string& detail = "", Args&&... args) {
        return Status(ErrParameterMissing,
                      "missing parameter",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    MetricNotFound(const std::string& detail = "", Args&&... args) {
        return Status(ErrMetricNotFound,
                      "metric not found",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    FieldNotFound(const std::string& detail = "", Args&&... args) {
        return Status(ErrFieldNotFound,
                      "field not found",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    FieldInvalidName(const std::string& detail = "", Args&&... args) {
        return Status(ErrFieldInvalidName,
                      "field name invalid",
                      string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    Undefined(const std::string& detail = "", Args&&... args) {
        return Status(ErrUndefined, "", string_util::SFormat(detail, args...));
    }

    template <class... Args>
    static Status
    Ok(const std::string& detail = "", Args&&... args) {
        return Status(0, "", string_util::SFormat(detail, args...));
    }

 public:
    bool
    IsOk() {
        return code_ == 0;
    }

    bool
    IsErr() {
        return code_ != 0;
    }

    int
    Code() {
        return code_;
    }

    const std::string&
    Msg() {
        return msg_;
    }

    const std::string&
    Detail() {
        return detail_;
    }

 private:
    int code_;
    std::string msg_;
    std::string detail_;
};

inline Status&
Status::operator=(Status&& rhs) {
    code_ = rhs.code_;
    msg_ = std::move(rhs.msg_);
    detail_ = std::move(rhs.detail_);
    return *this;
}

}  // namespace milvus::local
