#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { InfrastructureStack } from '../lib/infrastructure-stack';

const branchName = process.env.BRANCH_NAME || 'main';

const account = process.env.CDK_DEFAULT_ACCOUNT || 'ACCOUNT_ID';
const region = process.env.CDK_DEFAULT_REGION || 'REGION';

const app = new cdk.App();
new InfrastructureStack(app, `sagemaker-finetune-text-to-image-${branchName}`, {
    branchName,
    region,
    env: {
        account,
        region,
    },
});