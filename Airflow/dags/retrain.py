from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.exceptions import AirflowSkipException

from datetime import datetime,timedelta
import os

class FileSensorWithSkip(FileSensor):
    def poke(self, context):
        is_exist = super().poke(context)
        if not is_exist:
            raise AirflowSkipException(f"The file '{self.filepath}' does not exist.")
        return is_exist
    

default_args = {
    # 'owner': 'kyle',
    'depends_on_past': False,  # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정. False는 과거의 실행 결과 상관없이 매일 실행한다
    'start_date': datetime(2023,7,25,15),
    # 'retires': 1,  # 실패시 재시도 횟수
    # 'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
    # 'priority_weight': 10 # DAG의 우선 순위를 설정할 수 있음
    # 'end_date': datetime(2022, 4, 24) # DAG을 마지막으로 실행할 Date
    'execution_timeout': timedelta(hours=10), # 실행 타임아웃 : 300초 넘게 실행되면 종료
    # 'on_failure_callback': some_function # 만약에 Task들이 실패하면 실행할 함수
    # 'on_success_callback': some_other_function
    # 'on_retry_callback': another_function
    'max_active_runs' : 1, # 동시에 dag 실행 개수
}

# Ariflow/trigger 폴더 경로의 retrain.txt
trigger_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'trigger','retrain.txt')

with DAG(
    dag_id = 'retrain_dag',         # dag 
    default_args = default_args,    # dag 기본 옵션
    schedule_interval= "0 15 * * *", # 매 자정마다 실행하겠다 UTC 기준 15시
    tags=['retrain_dags'],
    catchup=False,
) as dag:
    
    # trigger file 확인, 없다면 skip
    file_check_task = FileSensorWithSkip(
        task_id='check_retrain_trigger',
        filepath=trigger_path,
        dag=dag,
    )
    
    # trigger file 삭제
    remove_file_task = BashOperator(
        task_id = 'delate_retrain_trigger',
        bash_command=f'rm {trigger_path}'
    )
    
    # github mlflow의 train 코드
    bash_command = f"python {os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/MLflow/train.py"
    
    retrain_task = BashOperator(
        task_id = 'retrain',
        bash_command=bash_command,
        dag=dag,
    )
    
    # trigger 파일 발견하면 삭제하고 retrain
    file_check_task >> retrain_task >> remove_file_task 