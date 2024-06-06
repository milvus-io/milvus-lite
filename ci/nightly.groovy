#!/usr/bin/env groovy

int total_timeout_minutes = 60 * 5

pipeline {
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
                            milvusdb/milvus-env:lite-main bash ci/entrypoint.sh
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
