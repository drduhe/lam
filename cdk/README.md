# LAM — AWS CDK (SageMaker)

This app deploys a **VPC**, **SageMaker execution role** (in a separate stack for ENI lifecycle), and a **SageMaker real-time endpoint** that runs the LAM image built from `docker/Dockerfile.lam-sagemaker` at the repository root.

## Prerequisites

- Node.js 18+ and npm
- AWS credentials for the target account
- Docker (when `BUILD_FROM_SOURCE` is `true`, for `cdk deploy` asset publishing)
- **Weights**: the Dockerfile expects `assets/weights/sam3.pt` (see `docker/README.md`).

## Configure

```bash
cd cdk
cp bin/deployment/deployment.json.example bin/deployment/deployment.json
```

Edit `bin/deployment/deployment.json`:

- Set `account.id` to your **12-digit AWS account ID** (must match the account in your credentials).
- Set `account.region`.
- Adjust `modelEndpointConfig` as needed (`INSTANCE_TYPE`, `MODEL_NAME` for the endpoint name, etc.).

When building from the repo (default), `CONTAINER_BUILD_PATH` is **`..`** (parent of `cdk/`, i.e. the LAM repo root) and `CONTAINER_DOCKERFILE` is **`docker/Dockerfile.lam-sagemaker`**.

To use a pre-pushed image instead, set `BUILD_FROM_SOURCE` to `false` and set `CONTAINER_URI` to your ECR or registry URI.

## Commands

```bash
npm install
npm run lint
npm test
npx cdk bootstrap aws://ACCOUNT_ID/REGION   # once per account/region
npx cdk deploy --all
```

Stack names default to `{projectName}-SageMakerRole`, `{projectName}-Network`, `{projectName}-Dataplane`.

## Notes

- **`.dockerignore`**: the repo root `.dockerignore` excludes `cdk/` so Docker/CDK asset staging does not pull in `cdk.out` or `node_modules` (which can cause very long paths or huge contexts).
- **VPC synthesis**: new VPCs use `maxAzs` + `RegionalConfig`; `cdk synth` may call AWS (for example AZ lookup) unless context is cached—use credentials that match `deployment.json`, or import an existing VPC via `networkConfig.VPC_ID` / `TARGET_SUBNETS`.
- **`deployment.json`** is gitignored; only `deployment.json.example` is tracked.
