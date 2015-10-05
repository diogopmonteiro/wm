from fabric.api import local


def pull_r():
    local('git pull --rebase origin master')


def commit(msg):
    # fab commit:"msg"
    if msg[0] == '"':
        msg = msg[1:]
    if msg[-1] == '"':
        msg = msg[:-1]

    local('git add --all')
    local('git commit -m \"' + msg + '\"')


def push():
    local('git push origin master')


def git_all(msg):
	commit(msg)
	pull_r()
	push()