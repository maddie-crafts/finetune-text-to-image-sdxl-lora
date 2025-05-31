import * as cdk from 'aws-cdk-lib';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';
import * as ecr_assets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecr_deployment from 'cdk-ecr-deployment';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';


interface ECRImageBuildProps {
    localImagePath: string;
    ecrRepositoryName: string;
    cluster: ecs.ICluster;
    targetBucket: s3.IBucket;
    aws_region: string;
    taskdefinitionName: string;
    cpu: number;
    memoryLimit: number;
    reuseExistingResources?: boolean;
}

export class ECRImageBuildAndTaskDefinition extends Construct {
    public readonly taskDefinition: ecs.FargateTaskDefinition;
    public readonly repository: ecr.IRepository;
    constructor(scope: Construct, id: string, props: ECRImageBuildProps) {
        super(scope, id);

    // Create task execution role with required permissions
    const executionRole = new iam.Role(this, 'EcsTaskExecutionRole', {
        assumedBy: new iam.CompositePrincipal(
          new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
          new iam.ServicePrincipal('sagemaker.amazonaws.com')
        ),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
          iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLambda_FullAccess')
        ],
      });
  
    const repo = props.reuseExistingResources
      ? ecr.Repository.fromRepositoryName(this, 'EcrRepo', props.ecrRepositoryName)
      : new ecr.Repository(this, 'EcrRepo', {
          repositoryName: props.ecrRepositoryName,
          imageScanOnPush: false,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
          emptyOnDelete: true,
        });

    // Build the Docker image
    const appImageAsset = new ecr_assets.DockerImageAsset(this,'DockerImageAsset', {
        directory: props.localImagePath,
        platform: ecr_assets.Platform.LINUX_AMD64
        });

    const imageDeployment = new ecr_deployment.ECRDeployment(this, 'DeployDockerImage', {
        src: new ecr_deployment.DockerImageName(appImageAsset.imageUri),
        dest: new ecr_deployment.DockerImageName(`${repo.repositoryUri}:latest`),
      });

    const taskRole = new iam.Role(this, 'TaskRole', {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'), 
          iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLambda_FullAccess')
        ],
      });

    props.targetBucket.grantWrite(taskRole);

    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          "logs:DescribeLogStreams",
          "logs:DescribeLogGroups",
          'cloudwatch:PutMetricData',
        ],
        resources: ['*'],
      }));
 
    const repoLogGroup = props.reuseExistingResources
      ? logs.LogGroup.fromLogGroupName(this, 'LogGroup', `/aws/ecs/${props.taskdefinitionName}`)
      : new logs.LogGroup(this, 'LogGroup', {
          logGroupName: `/aws/ecs/${props.taskdefinitionName}`,
          retention: logs.RetentionDays.ONE_DAY,
          removalPolicy: cdk.RemovalPolicy.RETAIN,
        });
   
    const TaskDef = this.createTaskDefinition(repoLogGroup, props.ecrRepositoryName, props.aws_region, appImageAsset, props.taskdefinitionName, executionRole, taskRole, repo, props.targetBucket, props.memoryLimit, props.cpu);
    TaskDef.node.addDependency(imageDeployment);
    this.taskDefinition = TaskDef;
    this.repository = repo;
    }

    private createTaskDefinition(logGroup: logs.ILogGroup, ecrRepositoryName: string, aws_region: string, asset: ecr_assets.DockerImageAsset, taskdefinitionName: string, executionRole: iam.IRole, taskRole: iam.IRole, repo: ecr.IRepository, targetBucket: s3.IBucket, memoryLimit: number, cpu: number): ecs.FargateTaskDefinition {
        const taskDef = new ecs.FargateTaskDefinition(this, 'TaskDef', {
          family: taskdefinitionName,
          memoryLimitMiB: memoryLimit,
          cpu: cpu,
          executionRole: executionRole,
          taskRole: taskRole,
          runtimePlatform: {
            cpuArchitecture: ecs.CpuArchitecture.X86_64,
          },
        });
    
        const container = taskDef.addContainer('AppContainer', {
          image: ecs.ContainerImage.fromEcrRepository(repo, 'latest'),
          environment: {
            BUCKET_NAME: targetBucket.bucketName,
            AWS_REGION: aws_region
          },
          logging: ecs.LogDrivers.awsLogs({
            streamPrefix: ecrRepositoryName,
            logGroup: logGroup,
            mode: ecs.AwsLogDriverMode.NON_BLOCKING,
          }),
    
        });

        container.addPortMappings(
          { containerPort: 8080, protocol: ecs.Protocol.TCP}
        );
    
        taskDef.addToTaskRolePolicy(new iam.PolicyStatement({
          actions: [
            'ecr:GetDownloadUrlForLayer',
            'ecr:BatchGetImage',
            'ecr:BatchCheckLayerAvailability',
            'ecr:GetRepositoryPolicy',
            'ecr:DescribeRepositories',
            'ecr:ListImages',
            'ecr:DescribeImages',
          ],
          resources: [repo.repositoryArn]
        }));
        taskDef.addToTaskRolePolicy(new iam.PolicyStatement({
          actions: ['ecr:GetAuthorizationToken'],
          resources: ['*'],
        }));
    
        return taskDef 

      }
  }