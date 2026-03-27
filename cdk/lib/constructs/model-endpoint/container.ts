import {
  ContainerImage as SageMakerContainerImage,
  ContainerImageConfig,
  Model,
} from "@aws-cdk/aws-sagemaker-alpha";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { ContainerImage } from "aws-cdk-lib/aws-ecs";
import { Construct } from "constructs";

class RegistryContainerImage extends SageMakerContainerImage {
  private readonly imageUri: string;

  constructor(imageUri: string) {
    super();
    this.imageUri = imageUri;
  }

  bind(scope: Construct, model: Model): ContainerImageConfig {
    void scope;
    void model;
    return {
      imageName: this.imageUri,
    };
  }
}

export class ContainerConfig {
  public readonly CONTAINER_URI: string;
  public readonly CONTAINER_BUILD_PATH?: string;
  public readonly CONTAINER_BUILD_TARGET?: string;
  public readonly CONTAINER_DOCKERFILE?: string;
  public readonly CONTAINER_BUILD_ARGS?: Record<string, string>;

  constructor(config?: Partial<ContainerConfig>) {
    this.CONTAINER_URI = config?.CONTAINER_URI ?? "lam/lam-sagemaker:latest";
    this.CONTAINER_BUILD_PATH = config?.CONTAINER_BUILD_PATH ?? "..";
    this.CONTAINER_DOCKERFILE =
      config?.CONTAINER_DOCKERFILE ?? "docker/Dockerfile.lam-sagemaker";
    this.CONTAINER_BUILD_TARGET = config?.CONTAINER_BUILD_TARGET;
    this.CONTAINER_BUILD_ARGS = config?.CONTAINER_BUILD_ARGS;
  }
}

export interface ServingContainerProps {
  account: {
    id: string;
    region: string;
    prodLike?: boolean;
  };
  buildFromSource: boolean;
  config: ContainerConfig;
}

export class ServingContainer extends Construct {
  public readonly dockerImageAsset?: DockerImageAsset;
  public readonly containerImage: ContainerImage;
  public readonly sagemakerContainerImage: SageMakerContainerImage;
  public readonly containerUri: string;
  public readonly repositoryAccessMode: string;

  constructor(scope: Construct, id: string, props: ServingContainerProps) {
    super(scope, id);

    if (props.buildFromSource) {
      this.sagemakerContainerImage = SageMakerContainerImage.fromAsset(
        props.config.CONTAINER_BUILD_PATH!,
        {
          file: props.config.CONTAINER_DOCKERFILE,
          target: props.config.CONTAINER_BUILD_TARGET,
          buildArgs: props.config.CONTAINER_BUILD_ARGS,
          platform: Platform.LINUX_AMD64,
        },
      );

      this.dockerImageAsset = new DockerImageAsset(this, "DockerAsset", {
        directory: props.config.CONTAINER_BUILD_PATH!,
        file: props.config.CONTAINER_DOCKERFILE,
        target: props.config.CONTAINER_BUILD_TARGET,
        buildArgs: props.config.CONTAINER_BUILD_ARGS,
        platform: Platform.LINUX_AMD64,
      });

      this.containerImage = ContainerImage.fromDockerImageAsset(
        this.dockerImageAsset,
      );
      this.containerUri = this.dockerImageAsset.imageUri;
      this.repositoryAccessMode = "Platform";
    } else {
      this.containerImage = ContainerImage.fromRegistry(
        props.config.CONTAINER_URI,
      );
      this.sagemakerContainerImage = new RegistryContainerImage(
        props.config.CONTAINER_URI,
      );
      this.containerUri = props.config.CONTAINER_URI;
      this.repositoryAccessMode = "Platform";
    }
  }
}
