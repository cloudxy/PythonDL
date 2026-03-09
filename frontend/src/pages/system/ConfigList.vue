<template>
  <div class="space-y-6">
    <PageHeader title="系统配置" subtitle="管理系统全局配置项" />

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '系统配置' }
    ]" />

    <!-- 配置分类标签 -->
    <Tabs v-model="activeTab" :tabs="configTabs" />

    <!-- 基础配置 -->
    <div v-show="activeTab === 'basic'">
      <Card title="基础配置">
        <FormSection title="网站信息" description="配置网站基本信息">
          <Input v-model="config.siteName" label="网站名称" placeholder="请输入网站名称" />
          <Input v-model="config.siteUrl" label="网站地址" placeholder="请输入网站地址" />
          <Textarea v-model="config.siteDescription" label="网站描述" placeholder="请输入网站描述" />
          <Input v-model="config.siteKeywords" label="SEO关键词" placeholder="多个关键词用逗号分隔" />
        </FormSection>

        <FormSection title="联系信息">
          <Input v-model="config.contactEmail" label="联系邮箱" placeholder="请输入联系邮箱" />
          <Input v-model="config.contactPhone" label="联系电话" placeholder="请输入联系电话" />
          <Input v-model="config.contactAddress" label="联系地址" placeholder="请输入联系地址" />
        </FormSection>

        <template #footer>
          <div class="flex justify-end">
            <Button variant="primary" :loading="saving" @click="saveConfig">保存配置</Button>
          </div>
        </template>
      </Card>
    </div>

    <!-- 邮件配置 -->
    <div v-show="activeTab === 'email'">
      <Card title="邮件配置">
        <FormSection title="SMTP设置">
          <Input v-model="config.smtpHost" label="SMTP服务器" placeholder="如：smtp.qq.com" />
          <Input v-model="config.smtpPort" type="number" label="端口" placeholder="如：465" />
          <Input v-model="config.smtpUser" label="用户名" placeholder="邮箱地址" />
          <PasswordInput v-model="config.smtpPassword" label="密码" placeholder="邮箱密码或授权码" />
          <Input v-model="config.smtpFrom" label="发件人" placeholder="发件人名称" />
          <Select
            v-model="config.smtpEncryption"
            label="加密方式"
            :options="encryptionOptions"
          />
        </FormSection>

        <template #footer>
          <div class="flex justify-end gap-3">
            <Button variant="secondary" @click="testEmail">发送测试邮件</Button>
            <Button variant="primary" :loading="saving" @click="saveConfig">保存配置</Button>
          </div>
        </template>
      </Card>
    </div>

    <!-- 安全配置 -->
    <div v-show="activeTab === 'security'">
      <Card title="安全配置">
        <FormSection title="登录设置">
          <Input v-model="config.maxLoginAttempts" type="number" label="最大登录尝试次数" />
          <Input v-model="config.loginLockTime" type="number" label="锁定时间(分钟)" />
          <Input v-model="config.passwordMinLength" type="number" label="密码最小长度" />
          <Toggle v-model="config.requireSpecialChar" label="要求特殊字符" />
          <Toggle v-model="config.requireNumber" label="要求数字" />
          <Toggle v-model="config.requireUppercase" label="要求大写字母" />
        </FormSection>

        <FormSection title="会话设置">
          <Input v-model="config.sessionTimeout" type="number" label="会话超时时间(分钟)" />
          <Toggle v-model="config.allowMultiLogin" label="允许多端登录" />
        </FormSection>

        <template #footer>
          <div class="flex justify-end">
            <Button variant="primary" :loading="saving" @click="saveConfig">保存配置</Button>
          </div>
        </template>
      </Card>
    </div>

    <!-- 存储配置 -->
    <div v-show="activeTab === 'storage'">
      <Card title="存储配置">
        <FormSection title="文件上传">
          <Input v-model="config.maxFileSize" type="number" label="最大文件大小(MB)" />
          <Input v-model="config.allowedFileTypes" label="允许的文件类型" placeholder="如：jpg,png,pdf" />
          <Select
            v-model="config.storageDriver"
            label="存储方式"
            :options="storageOptions"
          />
        </FormSection>

        <FormSection title="OSS配置" v-if="config.storageDriver === 'oss'">
          <Input v-model="config.ossEndpoint" label="Endpoint" />
          <Input v-model="config.ossBucket" label="Bucket" />
          <Input v-model="config.ossAccessKey" label="Access Key" />
          <PasswordInput v-model="config.ossSecretKey" label="Secret Key" />
        </FormSection>

        <template #footer>
          <div class="flex justify-end">
            <Button variant="primary" :loading="saving" @click="saveConfig">保存配置</Button>
          </div>
        </template>
      </Card>
    </div>

    <!-- API配置 -->
    <div v-show="activeTab === 'api'">
      <Card title="API配置">
        <FormSection title="接口设置">
          <Input v-model="config.apiRateLimit" type="number" label="API请求限制(次/分钟)" />
          <Toggle v-model="config.apiLogEnabled" label="启用API日志" />
          <Toggle v-model="config.apiCorsEnabled" label="启用CORS" />
          <Input v-model="config.corsOrigins" label="允许的域名" placeholder="多个域名用逗号分隔" />
        </FormSection>

        <template #footer>
          <div class="flex justify-end">
            <Button variant="primary" :loading="saving" @click="saveConfig">保存配置</Button>
          </div>
        </template>
      </Card>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Textarea from '@/components/ui/Textarea.vue'
import PasswordInput from '@/components/ui/PasswordInput.vue'
import Select from '@/components/ui/Select.vue'
import Toggle from '@/components/ui/Toggle.vue'
import Tabs from '@/components/ui/Tabs.vue'
import FormSection from '@/components/ui/FormSection.vue'

const activeTab = ref('basic')
const saving = ref(false)

const configTabs = [
  { value: 'basic', label: '基础配置' },
  { value: 'email', label: '邮件配置' },
  { value: 'security', label: '安全配置' },
  { value: 'storage', label: '存储配置' },
  { value: 'api', label: 'API配置' }
]

const config = reactive({
  // 基础配置
  siteName: 'PythonDL 智能分析平台',
  siteUrl: 'https://pythondl.example.com',
  siteDescription: '集成多种功能模块的全栈智能分析平台',
  siteKeywords: 'Python,数据分析,机器学习',
  contactEmail: 'admin@example.com',
  contactPhone: '400-123-4567',
  contactAddress: '北京市海淀区',

  // 邮件配置
  smtpHost: 'smtp.qq.com',
  smtpPort: 465,
  smtpUser: '',
  smtpPassword: '',
  smtpFrom: 'PythonDL',
  smtpEncryption: 'ssl',

  // 安全配置
  maxLoginAttempts: 5,
  loginLockTime: 30,
  passwordMinLength: 8,
  requireSpecialChar: true,
  requireNumber: true,
  requireUppercase: false,
  sessionTimeout: 120,
  allowMultiLogin: false,

  // 存储配置
  maxFileSize: 10,
  allowedFileTypes: 'jpg,png,pdf,doc,docx',
  storageDriver: 'local',
  ossEndpoint: '',
  ossBucket: '',
  ossAccessKey: '',
  ossSecretKey: '',

  // API配置
  apiRateLimit: 100,
  apiLogEnabled: true,
  apiCorsEnabled: true,
  corsOrigins: '*'
})

const encryptionOptions = [
  { value: 'none', label: '无加密' },
  { value: 'ssl', label: 'SSL' },
  { value: 'tls', label: 'TLS' }
]

const storageOptions = [
  { value: 'local', label: '本地存储' },
  { value: 'oss', label: '阿里云OSS' },
  { value: 'cos', label: '腾讯云COS' },
  { value: 'qiniu', label: '七牛云' }
]

const saveConfig = async () => {
  saving.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    alert('配置保存成功')
  } finally {
    saving.value = false
  }
}

const testEmail = async () => {
  alert('测试邮件已发送，请检查收件箱')
}
</script>
