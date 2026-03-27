/**
 * AWS account and deployment context for CDK stacks.
 */
export interface DeploymentAccount {
  readonly id: string;
  readonly region: string;
  readonly prodLike?: boolean;
  readonly isAdc?: boolean;
}

export type ConfigType = Record<string, unknown>;

export abstract class BaseConfig {
  constructor(config: Partial<ConfigType> = {}) {
    Object.assign(this, config);
  }
}

export interface RegionalConfigType {
  s3Endpoint: string;
  maxVpcAzs: number;
}

export class RegionalConfig {
  private static readonly configs: Record<string, RegionalConfigType> = {
    "us-east-1": {
      s3Endpoint: "s3.amazonaws.com",
      maxVpcAzs: 3,
    },
    "us-west-2": {
      s3Endpoint: "s3.us-west-2.amazonaws.com",
      maxVpcAzs: 3,
    },
    "us-west-1": {
      s3Endpoint: "s3.us-west-1.amazonaws.com",
      maxVpcAzs: 2,
    },
    "eu-west-1": {
      s3Endpoint: "s3.eu-west-1.amazonaws.com",
      maxVpcAzs: 3,
    },
    "ap-southeast-1": {
      s3Endpoint: "s3.ap-southeast-1.amazonaws.com",
      maxVpcAzs: 3,
    },
    "us-gov-west-1": {
      s3Endpoint: "s3.us-gov-west-1.amazonaws.com",
      maxVpcAzs: 2,
    },
    "us-gov-east-1": {
      s3Endpoint: "s3.us-gov-east-1.amazonaws.com",
      maxVpcAzs: 2,
    },
  };

  static getConfig(region: string): RegionalConfigType {
    return this.configs[region] || this.configs["us-east-1"];
  }
}
