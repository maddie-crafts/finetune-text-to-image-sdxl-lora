import boto3
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline_context import PipelineSession

def create_boto_and_sagemaker_sessions(region: str = 'us-west-2'):
    boto_session = boto3.Session(region_name=region)
    sess = PipelineSession(boto_session=boto_session)
    return boto_session, sess

def get_role(boto_session):
    iam = boto_session.client('iam')
    try:
        return get_execution_role()
    except ValueError:
        try:
            return iam.get_role(RoleName='AmazonSageMaker-ExecutionRole')['Role']['Arn']
        except Exception as e:
            print(f"Error retrieving role: {e}")
            return None
