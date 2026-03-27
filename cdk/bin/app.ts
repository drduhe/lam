#!/usr/bin/env node

import { App, Stack } from "aws-cdk-lib";

import { SageMakerRole } from "../lib/constructs/model-endpoint/roles/sagemaker-role";
import { ModelEndpointStack } from "../lib/model-endpoint-stack";
import { NetworkStack } from "../lib/network-stack";
import { loadDeploymentConfig } from "./deployment/load-deployment";

function main(): void {
  const app = new App();
  const deployment = loadDeploymentConfig();

  const env = {
    account: deployment.account.id,
    region: deployment.account.region,
  };

  const sagemakerRoleStack = new Stack(
    app,
    `${deployment.projectName}-SageMakerRole`,
    {
      env: {
        account: deployment.account.id,
        region: deployment.account.region,
      },
    },
  );
  const sagemakerRole = new SageMakerRole(
    sagemakerRoleStack,
    `${deployment.projectName}-SageMakerRole`,
    {
      account: deployment.account,
      roleName: `${deployment.projectName}-SageMakerRole`,
    },
  );

  const networkStack = new NetworkStack(
    app,
    `${deployment.projectName}-Network`,
    {
      env,
      deployment,
    },
  );

  networkStack.node.addDependency(sagemakerRoleStack);

  const modelEndpointStack = new ModelEndpointStack(
    app,
    `${deployment.projectName}-Dataplane`,
    {
      env,
      deployment,
      vpc: networkStack.network.vpc,
      selectedSubnets: networkStack.network.selectedSubnets,
      securityGroup: networkStack.network.securityGroup,
      sagemakerRole: sagemakerRole.role,
    },
  );

  modelEndpointStack.addDependency(networkStack);
  modelEndpointStack.addDependency(sagemakerRoleStack);
}

main();
