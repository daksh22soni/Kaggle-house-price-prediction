pipeline {
    agent any

    environment {
        IMAGE_NAME = "house-price-app"
        CONTAINER_NAME = "house-price-container"
    }

    stages {

        stage('Clone Repo') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/priyanshunayak2503-hue/kaagle-house-price.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('Stop Old Container') {
            steps {
                bat 'docker stop %CONTAINER_NAME% || exit 0'
                bat 'docker rm %CONTAINER_NAME% || exit 0'
            }
        }

        stage('Deploy New Container') {
            steps {
                bat '''
                    docker run -d ^
                    --name %CONTAINER_NAME% ^
                    -p 8501:8501 ^
                    -p 8000:8000 ^
                    %IMAGE_NAME%
                '''
            }
        }

        stage('Health Check') {
            steps {
                bat 'ping -n 16 127.0.0.1 > nul'
                bat 'curl http://localhost:8501 || exit 0'
            }
        }
    }

    post {
        success {
            echo 'Deployment Successful'
        }
        failure {
            echo 'Deployment Failed'
        }
    }
}