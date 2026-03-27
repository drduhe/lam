import {
  ContainerImage,
  Endpoint,
  EndpointConfig,
  InstanceType,
  Model,
} from "@aws-cdk/aws-sagemaker-alpha";
import { ISecurityGroup, IVpc, SubnetSelection } from "aws-cdk-lib/aws-ec2";
import { IRole } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export class SageMakerEndpointConfig {
  public readonly INITIAL_INSTANCE_COUNT: number;
  public readonly INITIAL_VARIANT_WEIGHT: number;
  public readonly VARIANT_NAME: string;
  public readonly SECURITY_GROUP_ID: string;
  public readonly CONTAINER_ENV: Record<string, string>;
  public readonly REPOSITORY_ACCESS_MODE: string;

  constructor(config?: Partial<SageMakerEndpointConfig>) {
    this.INITIAL_INSTANCE_COUNT = config?.INITIAL_INSTANCE_COUNT ?? 1;
    this.INITIAL_VARIANT_WEIGHT = config?.INITIAL_VARIANT_WEIGHT ?? 1;
    this.VARIANT_NAME = config?.VARIANT_NAME ?? "AllTraffic";
    this.SECURITY_GROUP_ID = config?.SECURITY_GROUP_ID ?? "";
    this.CONTAINER_ENV = config?.CONTAINER_ENV ?? {};
    this.REPOSITORY_ACCESS_MODE = config?.REPOSITORY_ACCESS_MODE ?? "Platform";
  }
}

export interface SageMakerEndpointConstructProps {
  role: IRole;
  containerImage: ContainerImage;
  endpointName: string;
  instanceType: InstanceType;
  vpc: IVpc;
  subnetSelection?: SubnetSelection;
  securityGroups?: ISecurityGroup[];
  config?: SageMakerEndpointConfig;
}

export class SageMakerEndpointConstruct extends Construct {
  public readonly endpointConfig: EndpointConfig;
  public readonly endpoint: Endpoint;
  public readonly model: Model;

  constructor(
    scope: Construct,
    id: string,
    props: SageMakerEndpointConstructProps,
  ) {
    super(scope, id);

    const config = props.config ?? new SageMakerEndpointConfig();

    this.model = new Model(this, "Model", {
      containers: [
        {
          image: props.containerImage,
          environment: config.CONTAINER_ENV,
        },
      ],
      role: props.role,
      vpc: props.vpc,
      vpcSubnets: props.subnetSelection,
      securityGroups: props.securityGroups,
    });

    this.endpointConfig = new EndpointConfig(this, "EndpointConfig", {
      instanceProductionVariants: [
        {
          model: this.model,
          variantName: config.VARIANT_NAME,
          initialVariantWeight: config.INITIAL_VARIANT_WEIGHT,
          instanceType: props.instanceType,
          initialInstanceCount: config.INITIAL_INSTANCE_COUNT,
        },
      ],
    });

    const endpointName = props.endpointName.substring(0, 63);

    this.endpoint = new Endpoint(this, "Endpoint", {
      endpointName: endpointName,
      endpointConfig: this.endpointConfig,
    });
  }
}
