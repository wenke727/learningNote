# VSCODE

## Shortcut

* `ctrl + ~` open/ close the terminal

* 折叠代码
  `ctrl k -`
  `ctrl k +`
  `ctrl k j`
  `ctrl k 0`

* tab页的快捷键
  `ctrl + tab`
Code>Preferences>Keyboard Shortcut
`workbench.action.previousEditor` command+h
`workbench.action.nextEditor` command+l

## Settings

* enablePreview
  `workbench.editor.enablePreview` -> True/False

单击一个右侧侧边栏的文件是预览模式，如果不输入任何任何文本就始终保持预览模式。
预览模式是打开一个新文件，然后再打开一个新文件，第二个就会占用第一个窗口。详细信息可以查看：<https://code.visualstudio.com/docs/getstarted/userinterface#_preview-mode>

* 免密登录

  ```bash
  ssh-keygen
  ssh-copy-id -i ~/.ssh/id_rsa.pub root@192.168.235.22
  cd ~/.ssh
  vim authorized_keys
  ```

## 常见问题

* VS code `containes emphasized items` but no error
   solve the problem react-native start --reset-cache and by reloading VS with the command `Ctrl + shift + p` and searching Developer: `Reload Window`
