import { CfnOutput, Environment, Stack, StackProps } from "aws-cdk-lib";
import { IVpc } from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

import { DeploymentConfig } from "../bin/deployment/load-deployment";
import { Network, NetworkConfig } from "./constructs/model-endpoint/network";

export interface NetworkStackProps extends StackProps {
  env: Environment;
  deployment: DeploymentConfig;
  vpc?: IVpc;
}

export class NetworkStack extends Stack {
  public readonly network: Network;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);

    const { deployment } = props;

    const networkConfig = deployment.networkConfig
      ? new NetworkConfig(deployment.networkConfig as Record<string, unknown>)
      : new NetworkConfig();

    this.network = new Network(this, "Network", {
      account: {
        id: deployment.account.id,
        region: deployment.account.region,
        prodLike: deployment.account.prodLike,
        isAdc: deployment.account.isAdc,
      },
      config: networkConfig,
      vpc: props.vpc,
    });

    new CfnOutput(this, "VpcId", {
      value: this.network.vpc.vpcId,
      description: "VPC ID",
      exportName: `${deployment.projectName}-VpcId`,
    });

    new CfnOutput(this, "SecurityGroupId", {
      value: this.network.securityGroup.securityGroupId,
      description: "Security Group ID",
      exportName: `${deployment.projectName}-SecurityGroupId`,
    });
  }
}
