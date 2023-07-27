# AirFlow - Train Server

## 1. Install requirements

```
bash requirements.sh
```
만약 AirFlow path를 home이 아닌 다른 파일에서 실행시키고 싶은 경우
```
export AIRFLOW_HOME=path
bash requirements.sh
```


## 2. Start scheduler
웹 서버로 접속
```
airflow webserver -h 0.0.0.0 -p 12345
```

스케줄러 실행
```
airflow scheduler
```

## 3. 기능
- make_data DAG    
    MLflow/new_data안 new data를 데이터셋으로 변환하는 DAG    
    trigger 폴더 안 retrain trigger 생성    

- retrain DAG    
    mlflow를 사용하여 train 실행

