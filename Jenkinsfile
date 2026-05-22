pipeline {
    agent any

    parameters {
        choice(
            name: 'SKIP_TRAINING',
            choices: ['false', 'true'],
            description: 'Skip training and use existing model (for deployment only)'
        )

        string(
            name: 'MIN_ACCURACY',
            defaultValue: '0.70',
            description: 'Minimum model accuracy threshold for quality gate'
        )
    }

    environment {
        PROJECT_NAME = "loan-risk-prediction"
        DOCKER_IMAGE = "loan-risk-app"
        CONTAINER_NAME = "loan-risk-container"
        MLFLOW_PORT = "5000"
        API_PORT = "8000"
    }

    stages {

        stage('Checkout Code') {
            steps {
                echo "Cloning GitHub repository..."

                git branch: 'main',
                url: 'https://github.com/Sumitkalamkar/mlops-loan-risk-prediction.git'
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Setting up Python environment..."

                sh '''
                    apt-get update

                    apt-get install -y \
                    python3 \
                    python3-pip \
                    python3-venv \
                    git \
                    docker.io \
                    curl

                    python3 --version
                    pip3 --version

                    python3 -m venv venv

                    ./venv/bin/pip install --upgrade pip setuptools wheel

                    ./venv/bin/pip install -r requirements.txt --no-cache-dir

                    echo "Installed packages successfully"

                    ./venv/bin/python --version
                    ./venv/bin/dvc --version
                    ./venv/bin/mlflow --version
                '''
            }
        }

        stage('Pull Data from DVC Remote') {
            steps {
                echo "Pulling dataset and artifacts from S3..."

                sh '''
                    ./venv/bin/dvc pull
                '''
            }
        }

        stage('Start MLflow Server') {
            steps {
                echo "Starting MLflow server..."

                sh '''
                    nohup ./venv/bin/mlflow ui \
                    --host 0.0.0.0 \
                    --port ${MLFLOW_PORT} \
                    > mlflow.log 2>&1 &
                '''
            }
        }

        stage('Run ML Pipeline') {
            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Running DVC ML pipeline..."

                sh '''
                    ./venv/bin/dvc repro
                '''
            }
        }

        stage('Quality Gate') {
            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Checking model quality..."

                sh '''
                    python3 - <<EOF
import json
import sys

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics.get("accuracy", 0)

print(f"Model Accuracy: {accuracy}")

minimum = float("${MIN_ACCURACY}")

if accuracy < minimum:
    print(f"FAILED: Accuracy {accuracy} < {minimum}")
    sys.exit(1)

print(f"PASSED: Accuracy {accuracy} >= {minimum}")
EOF
                '''
            }
        }

        stage('Push Artifacts to S3') {
            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Pushing updated artifacts to S3..."

                sh '''
                    ./venv/bin/dvc push
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."

                sh '''
                    docker build -t ${DOCKER_IMAGE} .
                '''
            }
        }

        stage('Deploy Application') {
            steps {
                echo "Deploying FastAPI application..."

                sh '''
                    docker stop ${CONTAINER_NAME} || true

                    docker rm ${CONTAINER_NAME} || true

                    docker run -d \
                    --name ${CONTAINER_NAME} \
                    -p ${API_PORT}:8000 \
                    ${DOCKER_IMAGE}
                '''
            }
        }

        stage('Health Check') {
            steps {
                echo "Checking API health..."

                sh '''
                    sleep 15

                    curl http://localhost:${API_PORT}/docs
                '''
            }
        }
    }

    post {

        success {
            echo '''
===================================================
PIPELINE COMPLETED SUCCESSFULLY
===================================================

Application URL:
http://YOUR_EC2_PUBLIC_IP:8000/docs

MLflow URL:
http://YOUR_EC2_PUBLIC_IP:5000
===================================================
'''
        }

        failure {
            echo '''
===================================================
PIPELINE FAILED
===================================================
'''
        }

        always {
            echo "Collecting pipeline artifacts..."

            sh '''
                mkdir -p artifacts

                cp -r metrics.json artifacts/ || true

                cp -r dvc.lock artifacts/ || true

                cp -r models/*.pkl artifacts/ || true
            '''

            archiveArtifacts artifacts: 'artifacts/**', fingerprint: true
        }
    }
}
