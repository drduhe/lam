import { CfnOutput, Environment, Stack, StackProps } from "aws-cdk-lib";
import { ISecurityGroup, IVpc, SubnetSelection } from "aws-cdk-lib/aws-ec2";
import { IRole } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

import { DeploymentConfig } from "../bin/deployment/load-deployment";
import {
  Dataplane,
  DataplaneConfig,
} from "./constructs/model-endpoint/dataplane";

export interface ModelEndpointStackProps extends StackProps {
  env: Environment;
  deployment: DeploymentConfig;
  vpc: IVpc;
  selectedSubnets: SubnetSelection;
  securityGroup: ISecurityGroup;
  sagemakerRole: IRole;
}

export class ModelEndpointStack extends Stack {
  public readonly resources: Dataplane;
  public readonly vpc: IVpc;
  public readonly endpointName: string;

  constructor(scope: Construct, id: string, props: ModelEndpointStackProps) {
    super(scope, id, {
      terminationProtection: props.deployment.account.prodLike,
      ...props,
    });

    const { deployment, vpc, selectedSubnets, securityGroup, sagemakerRole } =
      props;

    this.vpc = vpc;

    const dataplaneConfig = deployment.modelEndpointConfig
      ? new DataplaneConfig(deployment.modelEndpointConfig)
      : undefined;

    this.resources = new Dataplane(this, "Dataplane", {
      account: deployment.account,
      vpc,
      securityGroup,
      subnetSelection: selectedSubnets,
      projectName: deployment.projectName,
      sagemakerRole,
      config: dataplaneConfig,
    });

    this.endpointName =
      this.resources.modelEndpoint.endpoint.endpointName || "";

    new CfnOutput(this, "EndpointName", {
      value: this.endpointName,
      description: "SageMaker Endpoint Name",
      exportName: `${deployment.projectName}-EndpointName`,
    });
  }
}
