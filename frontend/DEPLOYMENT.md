# PythonDL 前端部署说明

## 字体问题解决方案

由于 Google Fonts 在国内访问不稳定，项目已移除对 Google Fonts 的依赖，改用系统默认字体。

### 修改内容

1. **index.html** - 移除了 Google Fonts 引用
2. **tailwind.config.js** - 移除了 'Inter' 字体，使用系统字体栈

### 系统字体栈

现在使用以下系统字体栈，确保在所有平台上都有良好的显示效果：

```javascript
fontFamily: {
  sans: [
    'system-ui',
    '-apple-system',
    'BlinkMacSystemFont',
    'Segoe UI',
    'Roboto',
    'Helvetica Neue',
    'Arial',
    'Noto Sans',
    'sans-serif',
  ],
}
```

### 优势

- ✅ 无需加载外部字体资源
- ✅ 页面加载速度更快
- ✅ 无网络访问问题
- ✅ 系统原生渲染，性能更好

## 服务启动

### 开发环境

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

服务将运行在 http://localhost:3000

### 生产环境

```bash
# 构建
npm run build

# 预览
npm run preview
```

## 注意事项

如果之前访问过旧版本，请清除浏览器缓存：
- Chrome/Edge: `Ctrl+Shift+Delete` (Windows) 或 `Cmd+Shift+Delete` (Mac)
- 或者使用无痕模式访问
