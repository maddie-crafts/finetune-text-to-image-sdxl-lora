import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';

interface ClusterConstructProps {
  vpc: ec2.IVpc;
  clusterName?: string;
}

export class ClusterConstruct extends Construct {
  public readonly cluster: ecs.Cluster;

  constructor(scope: Construct, id: string, props: ClusterConstructProps) {
    super(scope, id);

    this.cluster = new ecs.Cluster(this, 'Cluster', {
      vpc: props.vpc,
      clusterName: props.clusterName,
    });
  }
}