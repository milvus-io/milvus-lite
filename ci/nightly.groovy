#!/usr/bin/env groovy

int total_timeout_minutes = 60 * 5


// When scheduling a job that gets automatically triggered by changes,
// you need to include a [cronjob] tag within the commit message.
String cron_timezone = 'TZ=Asia/Shanghai'
String cron_string = BRANCH_NAME == "2.4" ? "50 3 * * * " : ""

pipeline {
    triggers {
        cron """${cron_timezone}
            ${cron_string}"""
    }
    options {
        timestamps()
        timeout(time: total_timeout_minutes, unit: 'MINUTES')
        buildDiscarder logRotator(artifactDaysToKeepStr: '30')
        parallelsAlwaysFailFast()
        preserveStashes(buildCount: 5)
        disableConcurrentBuilds(abortPrevious: true)
    }
    agent {
            kubernetes {
                cloud '4am'
                inheritFrom 'milvus-e2e-4am'
                defaultContainer 'main'
                yamlFile 'ci/rte-build.yaml'
                customWorkspace '/home/jenkins/agent/workspace'
            }
    }
    environment {
        DOCKER_BUILDKIT = 1
    }

    stages {
        stage('Build') {
            steps {
                container('main') {
                    script {
                        sh '''

                        MIRROR_URL="https://docker-nexus-ci.zilliz.cc" ./ci/set_docker_mirror.sh
                        '''
                        sh '''
                         docker run --net=host  \
                            -e CONAN_USER_HOME=/root/  -v \$PWD:/root/milvus-lite -v /root/.conan:/root/.conan -w /root/milvus-lite  \
                            milvusdb/milvus-env:lite-manylinux2014 bash ci/entrypoint.sh
                            # the image milvusdb/milvus-env:lite-manylinux2014 is from scripts/Dockerfile
                         '''
                    }
                }
            }
        }
        stage('arhive Artifacts ') {
            steps {
                container('main') {
                    archiveArtifacts artifacts: 'python/dist/*.whl',
                     allowEmptyArchive: true,
                     fingerprint: true,
                     onlyIfSuccessful: true
                }
            }
        }
        stage('install wheel') {
            steps {
                container('pytest') {
                    sh '''
                  pip install ./python/dist/*.whl
                  '''
                }
            }
        }
        stage('Test') {
            steps {
                container('pytest') {
                    sh '''
                    bash ci/test.sh -m nightly
                  '''
                }
            }
        }
    }
    post {
        unsuccessful {
            container('jnlp') {
                dir('ci') {
                    script {
                        def authorEmail = sh(returnStdout: true, script: './get_author_email.sh ')
                        emailext subject: '$DEFAULT_SUBJECT',
                      body: '$DEFAULT_CONTENT',
                      recipientProviders: [developers(), culprits()],
                      replyTo: '$DEFAULT_REPLYTO',
                      to: "${authorEmail},devops@zilliz.com"
                    }
                }
            }
        }
    }
}
