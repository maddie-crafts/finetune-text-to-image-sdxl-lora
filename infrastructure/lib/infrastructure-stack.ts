import * as cdk from 'aws-cdk-lib';
import * as path from 'path';
import { Construct } from 'constructs';
import { DefaultVpcConstruct } from './constructs/default-vpc';
import { ClusterConstruct } from './constructs/ecs-cluster';
import { S3Bucket } from './constructs/s3-bucket';
import { ECRImageBuildAndTaskDefinition } from './constructs/ecr';

export interface InfrastructureStackProps extends cdk.StackProps {
    branchName: string;
    region: string;
  }
  
  export class InfrastructureStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props?: InfrastructureStackProps) {
      super(scope, id, props);
      const branchName = props?.branchName;
      const codebuildSrcDir = process.env.CODEBUILD_SRC_DIR;
      const projectName = `sagemaker-finetune-text-to-image-${branchName}`;
      const processingBucket = new S3Bucket(this, 'ProcessingBucket', {
          bucketName: `sagemaker-finetune-text-to-image-bucket-${branchName}`,
      });

      const region = props?.region;

      const defaultVpc = new DefaultVpcConstruct(this, 'DefaultVpc');

      const cluster = new ClusterConstruct(this, 'Cluster', {
        vpc: defaultVpc.vpc,
        clusterName: `${projectName}-cluster`,
      });

      const StartPipeline = new ECRImageBuildAndTaskDefinition(this, 'StartPipeline', {
        localImagePath: path.join(__dirname, '../../src/containers/start_tuning'),
        taskdefinitionName: `${projectName}-start-finetuning-pipeline`,
        ecrRepositoryName: `start-finetuning-pipeline-${branchName}`,
        cpu: 512,
        memoryLimit: 1024,
        cluster: cluster.cluster,
        targetBucket: processingBucket.bucket,
        aws_region: `${region}`,
        reuseExistingResources: false,
      }); 

    }
}