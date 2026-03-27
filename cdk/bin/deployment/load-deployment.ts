import { existsSync, readFileSync } from "fs";
import { join } from "path";

import { NetworkConfig } from "../../lib/constructs/model-endpoint/network";

export interface DeploymentConfig {
  projectName: string;
  account: {
    id: string;
    region: string;
    prodLike?: boolean;
    isAdc?: boolean;
  };
  networkConfig?: NetworkConfig;
  modelEndpointConfig?: Partial<Record<string, unknown>>;
  deployIntegrationTests?: boolean;
  integrationTestConfig?: Partial<Record<string, unknown>>;
}

export class DeploymentConfigError extends Error {
  constructor(
    message: string,
    // eslint-disable-next-line no-unused-vars
    public field?: string,
  ) {
    super(message);
    this.name = "DeploymentConfigError";
  }
}

export function validateStringField(
  value: unknown,
  fieldName: string,
  isRequired: boolean = true,
): string {
  if (value === undefined || value === null) {
    if (isRequired) {
      throw new DeploymentConfigError(
        `Missing required field: ${fieldName}`,
        fieldName,
      );
    }
    return "";
  }

  if (typeof value !== "string") {
    throw new DeploymentConfigError(
      `Field '${fieldName}' must be a string, got ${typeof value}`,
      fieldName,
    );
  }

  const trimmed = value.trim();
  if (isRequired && trimmed === "") {
    throw new DeploymentConfigError(
      `Field '${fieldName}' cannot be empty or contain only whitespace`,
      fieldName,
    );
  }

  return trimmed;
}

export function validateAccountId(accountId: string): string {
  if (!/^\d{12}$/.test(accountId)) {
    throw new DeploymentConfigError(
      `Invalid AWS account ID format: '${accountId}'. Must be exactly 12 digits.`,
      "account.id",
    );
  }
  return accountId;
}

export function validateRegion(region: string): string {
  if (!/^[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*$/.test(region)) {
    throw new DeploymentConfigError(
      `Invalid AWS region format: '${region}'. Must follow pattern like 'us-east-1', 'eu-west-2', etc.`,
      "account.region",
    );
  }
  return region;
}

export function validateVpcId(vpcId: string): string {
  if (!/^vpc-[a-f0-9]{8}(?:[a-f0-9]{9})?$/.test(vpcId)) {
    throw new DeploymentConfigError(
      `Invalid VPC ID format: '${vpcId}'. Must start with 'vpc-' followed by 8 or 17 hexadecimal characters.`,
      "networkConfig.VPC_ID",
    );
  }
  return vpcId;
}

export function validateSecurityGroupId(securityGroupId: string): string {
  if (!/^sg-[a-f0-9]{8}(?:[a-f0-9]{9})?$/.test(securityGroupId)) {
    throw new DeploymentConfigError(
      `Invalid security group ID format: '${securityGroupId}'. Must start with 'sg-' followed by 8 or 17 hexadecimal characters.`,
      "networkConfig.SECURITY_GROUP_ID",
    );
  }
  return securityGroupId;
}

export function loadDeploymentConfig(): DeploymentConfig {
  const deploymentPath = join(__dirname, "deployment.json");

  if (!existsSync(deploymentPath)) {
    throw new DeploymentConfigError(
      `Missing deployment.json file at ${deploymentPath}. Please create it by copying deployment.json.example`,
    );
  }

  let parsed: unknown;
  try {
    const rawContent = readFileSync(deploymentPath, "utf-8");
    parsed = JSON.parse(rawContent) as unknown;
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new DeploymentConfigError(
        `Invalid JSON format in deployment.json: ${error.message}`,
      );
    }
    throw new DeploymentConfigError(
      `Failed to read deployment.json: ${error instanceof Error ? error.message : "Unknown error"}`,
    );
  }

  if (!parsed || typeof parsed !== "object" || parsed === null) {
    throw new DeploymentConfigError(
      "deployment.json must contain a valid JSON object",
    );
  }

  const parsedObj = parsed as Record<string, unknown>;

  const projectName = validateStringField(parsedObj.projectName, "projectName");
  if (projectName.length === 0) {
    throw new DeploymentConfigError("projectName cannot be empty");
  }

  if (!parsedObj.account || typeof parsedObj.account !== "object") {
    throw new DeploymentConfigError(
      "Missing or invalid account section in deployment.json",
      "account",
    );
  }

  const accountObj = parsedObj.account as Record<string, unknown>;

  const accountId = validateAccountId(
    validateStringField(accountObj.id, "account.id"),
  );
  const region = validateRegion(
    validateStringField(accountObj.region, "account.region"),
  );

  let networkConfig: DeploymentConfig["networkConfig"] = undefined;
  if (
    parsedObj.networkConfig &&
    typeof parsedObj.networkConfig === "object" &&
    parsedObj.networkConfig !== null
  ) {
    const networkConfigData = parsedObj.networkConfig as Record<
      string,
      unknown
    >;

    if (networkConfigData.VPC_ID !== undefined) {
      validateVpcId(
        validateStringField(networkConfigData.VPC_ID, "networkConfig.VPC_ID"),
      );
    }

    if (networkConfigData.TARGET_SUBNETS !== undefined) {
      if (!Array.isArray(networkConfigData.TARGET_SUBNETS)) {
        throw new DeploymentConfigError(
          "Field 'networkConfig.TARGET_SUBNETS' must be an array",
          "networkConfig.TARGET_SUBNETS",
        );
      }
    }

    if (networkConfigData.SECURITY_GROUP_ID !== undefined) {
      validateSecurityGroupId(
        validateStringField(
          networkConfigData.SECURITY_GROUP_ID,
          "networkConfig.SECURITY_GROUP_ID",
        ),
      );
    }

    if (
      networkConfigData.VPC_ID &&
      (!networkConfigData.TARGET_SUBNETS ||
        !Array.isArray(networkConfigData.TARGET_SUBNETS) ||
        networkConfigData.TARGET_SUBNETS.length === 0)
    ) {
      throw new DeploymentConfigError(
        "When VPC_ID is provided, TARGET_SUBNETS must also be specified with at least one subnet ID",
        "networkConfig.TARGET_SUBNETS",
      );
    }

    networkConfig = new NetworkConfig(networkConfigData);
  }

  let modelEndpointConfig: DeploymentConfig["modelEndpointConfig"] = undefined;
  if (
    parsedObj.modelEndpointConfig &&
    typeof parsedObj.modelEndpointConfig === "object" &&
    parsedObj.modelEndpointConfig !== null
  ) {
    modelEndpointConfig = parsedObj.modelEndpointConfig as Record<
      string,
      unknown
    >;
  }

  let deployIntegrationTests: boolean = false;
  if (
    parsedObj.deployIntegrationTests !== undefined &&
    typeof parsedObj.deployIntegrationTests === "boolean"
  ) {
    deployIntegrationTests = parsedObj.deployIntegrationTests;
  }

  let integrationTestConfig: DeploymentConfig["integrationTestConfig"] =
    undefined;
  if (
    parsedObj.integrationTestConfig &&
    typeof parsedObj.integrationTestConfig === "object" &&
    parsedObj.integrationTestConfig !== null
  ) {
    integrationTestConfig = parsedObj.integrationTestConfig as Record<
      string,
      unknown
    >;
  }

  const validatedConfig: DeploymentConfig = {
    projectName,
    account: {
      id: accountId,
      region: region,
      prodLike: (accountObj.prodLike as boolean | undefined) ?? false,
      isAdc: (accountObj.isAdc as boolean | undefined) ?? false,
    },
    networkConfig,
    modelEndpointConfig,
    deployIntegrationTests,
    integrationTestConfig,
  };

  const globalObj = global as { __deploymentConfigLoaded?: boolean };
  if (!globalObj.__deploymentConfigLoaded) {
    console.log(
      `Using environment from deployment.json: projectName=${validatedConfig.projectName}, region=${validatedConfig.account.region}`,
    );
    globalObj.__deploymentConfigLoaded = true;
  }

  return validatedConfig;
}
