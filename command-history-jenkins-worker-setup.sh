[root@ip-172-31-11-51 /]# history
    1  cd /
    2  wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.rpm
    3  yum install jdk-21_linux-x64_bin.rpm -y
    4  yum install git -y
    5  mkdir /data
    6  curl -sO http://3.109.212.198:8080/jnlpJars/agent.jar
    7  java -jar agent.jar -url http://3.109.212.198:8080/ -secret d3edb0280cb6c8d684d3407a2479103af11b4562d020340f6878cbf8b880eb4f -name "ml-worker" -webSocket -workDir "/data" &
    8  yum install docker -y
    9  systemctl start docker
   10  ssh-keygen -t rsa
   11  cat /root/.ssh/id_rsa.pub
   12  cd data/
   13  ls
   14  cd workspace/
   15  cd ml-pipeline
   16  ls
   17  cd data
   18  ls
   19  cd processed/
   20  ls
   21  cat featured_data.csv 
   22  cd ..
   23  ls
   24  cd ..
   25  ls
   26  cd models
   27  ls
   28  cd ..
   29  cat dvc.lock
   30  docker images
   31  nohup ./venv/bin/mlflow server                     --backend-store-uri sqlite:///mlflow.db                     --default-artifact-root ./mlruns                     --host 0.0.0.0                     --port 5000 &
   32  cd /
   33  history