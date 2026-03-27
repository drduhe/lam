import { region_info } from "aws-cdk-lib";
import {
  Effect,
  IRole,
  ManagedPolicy,
  PolicyStatement,
  Role,
  ServicePrincipal,
} from "aws-cdk-lib/aws-iam";
import { NagSuppressions } from "cdk-nag";
import { Construct } from "constructs";

import { BaseConfig, ConfigType, DeploymentAccount } from "../../types";

export class SageMakerRoleConfig extends BaseConfig {
  public SM_ROLE_NAME?: string | undefined;

  constructor(config: ConfigType = {}) {
    super({
      ...config,
    });
  }
}

export interface SageMakerRoleProps {
  readonly account: DeploymentAccount;
  readonly roleName: string;
  readonly config?: SageMakerRoleConfig;
  readonly existingRole?: IRole;
}

export class SageMakerRole extends Construct {
  public readonly config: SageMakerRoleConfig;
  public readonly role: IRole;
  public readonly partition: string;

  constructor(scope: Construct, id: string, props: SageMakerRoleProps) {
    super(scope, id);

    this.config = props.config ?? new SageMakerRoleConfig();

    this.partition = region_info.Fact.find(
      props.account.region,
      region_info.FactName.PARTITION,
    )!;

    if (this.config.SM_ROLE_NAME) {
      this.role = Role.fromRoleName(
        this,
        "ImportedSageMakerRole",
        this.config.SM_ROLE_NAME,
        {
          mutable: false,
        },
      );
    } else if (props.existingRole) {
      this.role = props.existingRole;
    } else {
      this.role = this.createSageMakerRole(props);
    }
  }

  private createSageMakerRole(props: SageMakerRoleProps): IRole {
    const role = new Role(this, "SageMakerExecutionRole", {
      roleName: props.roleName,
      assumedBy: new ServicePrincipal("sagemaker.amazonaws.com"),
      description:
        "Allows SageMaker to access necessary AWS services (S3, ECR, CloudWatch, VPC ENIs, ...)",
    });

    const smExecutionPolicy = new ManagedPolicy(
      this,
      "SageMakerExecutionPolicy",
      {
        managedPolicyName: `${props.roleName}-ExecutionPolicy`,
      },
    );

    const ec2NetworkPolicyStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeVpcEndpoints",
        "ec2:DescribeDhcpOptions",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterfacePermission",
        "ec2:DeleteNetworkInterface",
        "ec2:CreateNetworkInterfacePermission",
        "ec2:CreateNetworkInterface",
      ],
      resources: ["*"],
    });

    const ecrAuthPolicyStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ["ecr:GetAuthorizationToken"],
      resources: ["*"],
    });

    const ecrPolicyStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:DescribeImages",
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:DescribeRepositories",
      ],
      resources: [
        `arn:${this.partition}:ecr:${props.account.region}:${props.account.id}:repository/*`,
      ],
    });

    const cwLogsPolicyStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        "logs:CreateLogDelivery",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:DeleteLogDelivery",
        "logs:DescribeLogDeliveries",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams",
        "logs:DescribeResourcePolicies",
        "logs:GetLogEvents",
        "logs:GetLogDelivery",
        "logs:ListLogDeliveries",
        "logs:PutLogEvents",
        "logs:PutResourcePolicy",
        "logs:UpdateLogDelivery",
      ],
      resources: [
        `arn:${this.partition}:logs:${props.account.region}:${props.account.id}:log-group:*`,
      ],
    });

    const stsPolicyStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ["sts:AssumeRole"],
      resources: ["*"],
    });

    smExecutionPolicy.addStatements(
      cwLogsPolicyStatement,
      ecrAuthPolicyStatement,
      ecrPolicyStatement,
      ec2NetworkPolicyStatement,
      stsPolicyStatement,
    );

    role.addManagedPolicy(smExecutionPolicy);

    NagSuppressions.addResourceSuppressions(
      smExecutionPolicy,
      [
        {
          id: "AwsSolutions-IAM5",
          reason:
            "EC2 network interface actions require wildcard resource for VPC ENI management",
          appliesTo: ["Resource::*"],
        },
        {
          id: "AwsSolutions-IAM5",
          reason:
            "ecr:GetAuthorizationToken requires wildcard resource per AWS documentation",
          appliesTo: ["Resource::*"],
        },
        {
          id: "AwsSolutions-IAM5",
          reason:
            "sts:AssumeRole may target dynamically chosen roles for cross-account access",
          appliesTo: ["Resource::*"],
        },
        {
          id: "AwsSolutions-IAM5",
          reason:
            "ECR repository wildcard for account-local model image repositories",
          appliesTo: [
            `Resource::arn:${this.partition}:ecr:${props.account.region}:${props.account.id}:repository/*`,
          ],
        },
        {
          id: "AwsSolutions-IAM5",
          reason: "CloudWatch Logs wildcard for SageMaker-created log groups",
          appliesTo: [
            `Resource::arn:${this.partition}:logs:${props.account.region}:${props.account.id}:log-group:*`,
          ],
        },
      ],
      true,
    );

    return role;
  }
}
