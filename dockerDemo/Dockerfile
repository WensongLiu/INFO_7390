FROM python

ARG cik 
ARG acc_no_test
ARG access_key_id
ARG access_secret_key
ARG bucket_name

#copy test.py and run it
RUN pip install pandas
COPY Case1_Problem1_1012_test_docker.py .
RUN python3 Case1_Problem1_1012_test_docker.py, cik, acc_no_test, access_key_id, access_secret_key, bucket_name