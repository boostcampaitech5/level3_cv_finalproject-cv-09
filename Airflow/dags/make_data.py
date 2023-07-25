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
    'start_date': datetime(2023,7,25,14),
    # 'retires': 1,  # 실패시 재시도 횟수
    # 'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
    # 'priority_weight': 10 # DAG의 우선 순위를 설정할 수 있음
    # 'end_date': datetime(2022, 4, 24) # DAG을 마지막으로 실행할 Date
    'execution_timeout': timedelta(hours=1), # 실행 타임아웃 : 300초 넘게 실행되면 종료
    # 'on_failure_callback': some_function # 만약에 Task들이 실패하면 실행할 함수
    # 'on_success_callback': some_other_function
    # 'on_retry_callback': another_function
    'max_active_runs' : 1, # 동시에 dag 실행 개수
}

# Ariflow/trigger 폴더 경로의 retrain.txt
trigger_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'trigger','retrain.txt')

with DAG(
    dag_id = 'make_data_dag',         # dag 
    default_args = default_args,    # dag 기본 옵션
    schedule_interval= "0 14 * * *", # 매 자정마다 실행하겠다 UTC 기준 15시
    tags=['make_data_dags'],
    catchup=False,
) as dag:
    
    # github airflow의 make_data 코드
    bash_command = f"python {os.path.abspath(os.path.dirname(__file__))}/func/data.py"
    
    make_data_task = BashOperator(
        task_id = 'make_dataset',
        bash_command=bash_command,
        dag=dag,
    )
    
    # trigger file 생성
    remove_file_task = BashOperator(
        task_id = 'create_retrain_trigger',
        bash_command=f'cat > {trigger_path}'
    )
    
    # trigger 파일 발견하면 삭제하고 retrain
    make_data_task >> remove_file_task