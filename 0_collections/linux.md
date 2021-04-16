# Amax server

## Jupyter

* AI learing: http://192.168.135.34:8888/tree?
* Pytorch: http://192.168.135.34:8888/tree?

## 操作命令
* tar 压缩
```
tar cvf panos_test_dataset.tar  ./dataset
```
* Github
    * check the lines
        ```
        git log --format='%aN' | sort -u | while read name; do echo -en "$name\t"; git log --author="$name" --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\n", add, subs, loc }' -; done
        ```
