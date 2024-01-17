FROM milvusdb/milvus-env:amd64-centos7-20230606-c9d9940

# Ignore tool missing warnings, and wo also not need clang-tools
RUN rm -fr /etc/profile.d/llvm-toolset*.sh

# python38 for dev
RUN yum -y install rh-python38
RUN echo 'source scl_source enable rh-python38' > /etc/profile.d/rh-python38.sh

# git new for devel
RUN yum -y install rh-git227-git-all
RUN echo 'source scl_source enable rh-git227' > /etc/profile.d/rh-git227.sh

# Add local user in container for dev
RUN yum -y install sudo
ADD setup/env /tmp/env
RUN . /tmp/env && \
    if getent group $DEV_GROUP 2>/dev/null 1>/dev/null ; then \
        echo group $DEV_GROUP already exist ; \
    elif getent group $DEV_GID 2>/dev/null 1>/dev/null ; then \
        echo group $DEV_GID already exist ; \
    else \
        groupadd -g $DEV_GID $DEV_GROUP ; \
    fi ; \
    if id $DEV_USER 2>/dev/null 1>/dev/null ; then \
        echo user $DEV_USER already exist ; \
    elif id $DEV_UID 2>/dev/null 1>/dev/null ; then \
        echo user $DEV_UID already exist ; \
    else \
        useradd -g $DEV_GID -u $DEV_UID -m -d $DEV_HOME -s /bin/bash $DEV_USER ; \
    fi && \
    if [ $DEV_UID -ne 0 ] ; then \
        echo "${DEV_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user ; \
    fi
