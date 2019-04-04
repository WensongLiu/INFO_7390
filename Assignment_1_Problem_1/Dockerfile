FROM python
RUN pip install pandas
RUN pip install requests-html
RUN pip install boto3

ARG cik
ARG acc_no_test
ARG access_key_id
ARG access_secret_key
ARG bucket_name

COPY Case1_Problem1_test_docker_1.py .
RUN python Case1_Problem1_test_docker_1.py $cik $acc_no_test $access_key_id $access_secret_key $bucket_name