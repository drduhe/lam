import { InstanceType } from "@aws-cdk/aws-sagemaker-alpha";
import { RemovalPolicy } from "aws-cdk-lib";
import {
  ISecurityGroup,
  IVpc,
  SecurityGroup,
  SubnetSelection,
} from "aws-cdk-lib/aws-ec2";
import { IRole } from "aws-cdk-lib/aws-iam";
import { NagSuppressions } from "cdk-nag";
import { Construct } from "constructs";

import { BaseConfig, ConfigType, DeploymentAccount } from "../types";
import { ContainerConfig, ServingContainer } from "./container";
import {
  SageMakerEndpointConfig,
  SageMakerEndpointConstruct,
} from "./sagemaker-endpoint";

export class DataplaneConfig extends BaseConfig {
  public readonly BUILD_FROM_SOURCE: boolean;
  public readonly CONTAINER_URI: string;
  public readonly CONTAINER_BUILD_PATH: string;
  public readonly CONTAINER_BUILD_TARGET?: string;
  public readonly CONTAINER_DOCKERFILE: string;
  public readonly CONTAINER_BUILD_ARGS?: Record<string, string>;
  public readonly INSTANCE_TYPE: string;
  public readonly MODEL_NAME: string;
  public readonly CONTAINER_ENV: Record<string, string>;
  public readonly INITIAL_INSTANCE_COUNT: number;
  public readonly INITIAL_VARIANT_WEIGHT: number;
  public readonly VARIANT_NAME: string;
  public readonly SAGEMAKER_ROLE_NAME?: string;
  public readonly SECURITY_GROUP_ID?: string;

  constructor(config: Partial<ConfigType> = {}) {
    const mergedConfig = {
      BUILD_FROM_SOURCE: true,
      CONTAINER_URI: "lam/lam-sagemaker:latest",
      CONTAINER_BUILD_PATH: "..",
      CONTAINER_DOCKERFILE: "docker/Dockerfile.lam-sagemaker",
      INSTANCE_TYPE: "ml.g5.4xlarge",
      MODEL_NAME: "lam",
      CONTAINER_ENV: {} as Record<string, string>,
      INITIAL_INSTANCE_COUNT: 1,
      INITIAL_VARIANT_WEIGHT: 1,
      VARIANT_NAME: "AllTraffic",
      ...config,
    };
    super(mergedConfig);
    this.validateConfig(mergedConfig);
  }

  private validateConfig(config: Record<string, unknown>): void {
    const errors: string[] = [];

    const instanceCount =
      typeof config.INITIAL_INSTANCE_COUNT === "number"
        ? config.INITIAL_INSTANCE_COUNT
        : 1;
    if (instanceCount < 1) {
      errors.push("INITIAL_INSTANCE_COUNT must be at least 1");
    }

    const variantWeight =
      typeof config.INITIAL_VARIANT_WEIGHT === "number"
        ? config.INITIAL_VARIANT_WEIGHT
        : 1;
    if (variantWeight < 0 || variantWeight > 1) {
      errors.push("INITIAL_VARIANT_WEIGHT must be between 0 and 1");
    }

    const modelName = config.MODEL_NAME as string;
    if (!modelName || modelName.trim() === "") {
      errors.push("MODEL_NAME must be a non-empty string");
    }

    if (errors.length > 0) {
      throw new Error(`Configuration validation failed:\n${errors.join("\n")}`);
    }
  }
}

export interface DataplaneProps {
  readonly account: DeploymentAccount;
  readonly vpc: IVpc;
  readonly securityGroup?: ISecurityGroup;
  readonly subnetSelection?: SubnetSelection;
  readonly projectName: string;
  config?: DataplaneConfig;
  sagemakerRole: IRole;
}

export class Dataplane extends Construct {
  public readonly config: DataplaneConfig;
  public readonly removalPolicy: RemovalPolicy;
  public readonly sagemakerRole: IRole;
  public readonly container: ServingContainer;
  public readonly modelEndpoint: SageMakerEndpointConstruct;
  public readonly securityGroup?: ISecurityGroup;

  constructor(scope: Construct, id: string, props: DataplaneProps) {
    super(scope, id);

    this.config = this.initializeConfig(props);
    this.removalPolicy = this.initializeRemovalPolicy(props);
    this.securityGroup = this.initializeSecurityGroup(props);
    this.sagemakerRole = props.sagemakerRole;

    this.container = this.createContainer(props);
    this.modelEndpoint = this.createSageMakerEndpoint(props);

    if (this.container.dockerImageAsset) {
      NagSuppressions.addResourceSuppressions(
        this.container.dockerImageAsset,
        [
          {
            id: "AwsSolutions-IAM4",
            reason:
              "AWS managed policies are used for CDK custom resource Lambda execution roles.",
            appliesTo: [
              "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            ],
          },
          {
            id: "AwsSolutions-L1",
            reason:
              "Lambda runtime versions for CDK asset custom resources are managed by the CDK version.",
          },
        ],
        true,
      );
    }
  }

  private createContainer(props: DataplaneProps): ServingContainer {
    const containerConfig = new ContainerConfig({
      CONTAINER_URI: this.config.CONTAINER_URI,
      CONTAINER_BUILD_PATH: this.config.CONTAINER_BUILD_PATH,
      CONTAINER_BUILD_TARGET: this.config.CONTAINER_BUILD_TARGET,
      CONTAINER_DOCKERFILE: this.config.CONTAINER_DOCKERFILE,
      CONTAINER_BUILD_ARGS: this.config.CONTAINER_BUILD_ARGS,
    });

    return new ServingContainer(this, "Container", {
      account: {
        id: props.account.id,
        region: props.account.region,
        prodLike: props.account.prodLike,
      },
      buildFromSource: this.config.BUILD_FROM_SOURCE,
      config: containerConfig,
    });
  }

  private createSageMakerEndpoint(
    props: DataplaneProps,
  ): SageMakerEndpointConstruct {
    if (this.container.dockerImageAsset) {
      this.container.dockerImageAsset.repository.grantPull(this.sagemakerRole);
    }

    const endpointConfig = new SageMakerEndpointConfig({
      INITIAL_INSTANCE_COUNT: this.config.INITIAL_INSTANCE_COUNT,
      INITIAL_VARIANT_WEIGHT: this.config.INITIAL_VARIANT_WEIGHT,
      VARIANT_NAME: this.config.VARIANT_NAME,
      SECURITY_GROUP_ID: this.securityGroup?.securityGroupId ?? "",
      CONTAINER_ENV: this.config.CONTAINER_ENV,
      REPOSITORY_ACCESS_MODE: this.container.repositoryAccessMode,
    });

    const instanceType = new InstanceType(this.config.INSTANCE_TYPE);

    const endpoint = new SageMakerEndpointConstruct(this, "SageMakerEndpoint", {
      role: this.sagemakerRole,
      containerImage: this.container.sagemakerContainerImage,
      endpointName: this.config.MODEL_NAME,
      instanceType: instanceType,
      vpc: props.vpc,
      subnetSelection: props.subnetSelection,
      securityGroups: this.securityGroup ? [this.securityGroup] : undefined,
      config: endpointConfig,
    });

    if (
      this.container.dockerImageAsset &&
      this.sagemakerRole.node.defaultChild
    ) {
      endpoint.model.node.addDependency(this.sagemakerRole);
    }

    return endpoint;
  }

  private initializeConfig(props: DataplaneProps): DataplaneConfig {
    if (props.config instanceof DataplaneConfig) {
      return props.config;
    }
    return new DataplaneConfig(
      (props.config as unknown as Partial<ConfigType>) ?? {},
    );
  }

  private initializeRemovalPolicy(props: DataplaneProps): RemovalPolicy {
    return props.account.prodLike
      ? RemovalPolicy.RETAIN
      : RemovalPolicy.DESTROY;
  }

  private initializeSecurityGroup(
    props: DataplaneProps,
  ): ISecurityGroup | undefined {
    if (this.config.SECURITY_GROUP_ID) {
      return SecurityGroup.fromSecurityGroupId(
        this,
        "ImportedSecurityGroup",
        this.config.SECURITY_GROUP_ID,
      );
    }
    return props.securityGroup;
  }
}
