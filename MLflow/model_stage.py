from mlflow.tracking import MlflowClient
import mlflow
import argparse

def get_arg():
    parser = argparse.ArgumentParser(description="change model staging")
    parser.add_argument("--regist_name", type=str, default="register_model")
    parser.add_argument("--stage", type=str, default='Staging')
    
    args = parser.parse_args()
    return args

def run(args):
    # MLflow 클라이언트 생성
    client = MlflowClient()
    
    # 마지막 버전 가져오기 -> return list
    last_version = client.get_latest_versions(name=args.regist_name)[0]
    
    # currnet stage
    model = client.get_model_version(
        name = args.regist_name,
        version = last_version.version
    )
    print(f"model '{args.regist_name}' is on {model.current_stage}")
    
    # register model stage 변경
    client.transition_model_version_stage(
        name=args.regist_name,
        stage = args.stage,
        version=last_version.version
        )
    
    # check stage
    model = client.get_model_version(
        name = args.regist_name,
        version = last_version.version
    )
    print(f"success changing '{model.current_stage}' stage")

if __name__=="__main__":
    args = get_arg()
    run(args)