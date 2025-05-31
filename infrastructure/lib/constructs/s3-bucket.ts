import { Bucket, BucketEncryption, BucketAccessControl } from 'aws-cdk-lib/aws-s3';
import { RemovalPolicy, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';

interface S3BucketProps {
  bucketName?: string;
}

export class S3Bucket extends Construct {
  public readonly bucket: Bucket;

  constructor(scope: Construct, id: string, props: S3BucketProps = {}) {
    super(scope, id);

    this.bucket = new Bucket(this, 'ProcessingBucket', {
      bucketName: props.bucketName,
      encryption: BucketEncryption.S3_MANAGED,
      enforceSSL: true, 
      blockPublicAccess: {
        blockPublicAcls: true,
        blockPublicPolicy: true,
        ignorePublicAcls: true,
        restrictPublicBuckets: true,
      },
      accessControl: BucketAccessControl.PRIVATE,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    new CfnOutput(this, 'ProcessingBucketName', {
      value: this.bucket.bucketName,
      description: 'The name of the processing S3 bucket',
    });

    new CfnOutput(this, 'ProcessingBucketArn', {
      value: this.bucket.bucketArn,
      description: 'The ARN of the processing S3 bucket',
    });
  }
}