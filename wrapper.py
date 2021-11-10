import argparse
import json
import requests
import subprocess


def update_status(job_uid, job_status):
    url = 'http://host.docker.internal:8081/api/v0/jobs/' + job_uid + '/status'
    data = {'status': job_status}
    return requests.patch(url, json=data).status_code


def main(command, kwargs):
    full_cmd = command + ' ' + kwargs
    process = subprocess.Popen(full_cmd.split(' '))
    process.communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--my_dict', type=json.loads)
    args = parser.parse_args()
    arg_dict = args.my_dict

    try:
        main(arg_dict['command'], arg_dict['kwargs'])
    except ValueError as err:
        print(err)
        update_status(arg_dict['uid'], "failed")
    update_status(arg_dict['uid'], "complete")
