My CONFIG.yaml for kubernetes.

Thanks to https://github.com/HanGuo97/tf-library/blob/0.10/TFLibrary/docs/kubernetes_and_docker_guide.md


```yaml
apiVersion: batch/v1
# Resource Type: Job, Deployment, Service, etc. Here we use Job
# https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
kind: Job
metadata:
  # The name of the job
  name: xiaopeng-test
spec:
  # https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#pod-backoff-failure-policy
  backoffLimit: 5
  template:
    spec:
      restartPolicy: OnFailure
      # Pull from Private Registry
      # https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/

      # to pull from a private repo, e.g. Docker Hub
      # https://stackoverflow.com/a/36974280

      # to pull from nvidia
      # kubectl create secret docker-registry SECRET_NAME \
      #     --docker-server=nvcr.io
      #     --docker-username="\$oauthtoken"
      #     --docker-password=PASSWORD
      #     --docker-email=EMAIL

      # imagePullSecrets:
      # - name: SECRET_NAME

      # Set the security context for a Pod
      # https://kubernetes.io/docs/tasks/configure-pod-container/security-context/#set-the-security-context-for-a-pod
      securityContext:
        runAsUser: 243337
        fsGroup: 100

      # Volume could be: emptyPath, hostPath, etc
      # hostPath mounts host directories into pod
      # https://kubernetes.io/docs/concepts/storage/volumes/#hostpath
      volumes:
      - name: home
        hostPath:
          # directory location on host
          path: /nas/longleaf/home/xiaopeng
          # following field is optional
          # type: Directory

      containers:
      - name: vincent-torch-test
        # e.g. nvcr.io/nvidia/tensorflow:18.07-py3
        image: vincentlu073/xiaopeng-torch:latest
        imagePullPolicy: Always
        resources:
          # https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/
          # You can specify GPU limits without specifying requests
          # because Kubernetes will use the limit as the request value by default.
          # requests:
            # cpu: CPU_REQUEST
            # memory: MEMORY_REQUEST
          limits:
            cpu: 2
            memory: 10G
            nvidia.com/gpu: 1

        volumeMounts:
        # name must match the volume name defined in volumes
        - name: home
          # mount path within the container
          mountPath: /nas/longleaf/home/xiaopeng

        # https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/
        command: ["/bin/bash"]
        args: ["-c", "while true; do echo hello; sleep 10;done"]
```
