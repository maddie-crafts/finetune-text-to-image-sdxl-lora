import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export class DefaultVpcConstruct extends Construct {
  public readonly vpc: ec2.IVpc;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.vpc = ec2.Vpc.fromLookup(this, 'DefaultVPC', {
      isDefault: true,
    });
  }
}