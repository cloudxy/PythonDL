<template>
  <div class="space-y-6">
    <div class="text-center">
      <h2 class="text-2xl font-bold text-secondary-900">找回密码</h2>
      <p class="text-secondary-500 mt-2">请输入您的邮箱地址，我们将发送重置密码链接</p>
    </div>

    <!-- 步骤1: 输入邮箱 -->
    <form v-if="step === 1" @submit.prevent="handleSendEmail" class="space-y-5">
      <Input
        v-model="form.email"
        type="email"
        label="邮箱地址"
        placeholder="请输入注册时使用的邮箱"
        :error="errors.email"
        required
      />

      <Alert v-if="sendError" variant="danger" :message="sendError" />

      <Button type="submit" variant="primary" block :loading="loading">
        发送重置链接
      </Button>
    </form>

    <!-- 步骤2: 邮件已发送 -->
    <div v-else-if="step === 2" class="space-y-5">
      <div class="text-center py-6">
        <div class="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <h3 class="text-lg font-semibold text-secondary-900 mb-2">邮件已发送</h3>
        <p class="text-secondary-500">
          我们已向 <span class="font-medium text-secondary-700">{{ form.email }}</span> 发送了密码重置链接
        </p>
        <p class="text-sm text-secondary-400 mt-2">
          请检查您的收件箱（包括垃圾邮件文件夹）
        </p>
      </div>

      <Button variant="outline" block @click="handleResend" :loading="resending">
        重新发送邮件
      </Button>

      <p class="text-center text-sm text-secondary-500">
        <router-link to="/login" class="text-primary-600 hover:text-primary-700 font-medium">
          返回登录
        </router-link>
      </p>
    </div>

    <!-- 步骤3: 设置新密码 -->
    <form v-else-if="step === 3" @submit.prevent="handleResetPassword" class="space-y-5">
      <Input
        v-model="form.email"
        type="email"
        label="邮箱地址"
        disabled
      />

      <PasswordInput
        v-model="form.newPassword"
        label="新密码"
        placeholder="请输入新密码"
        :error="errors.newPassword"
        hint="密码长度至少为8位，包含字母和数字"
        required
      />

      <PasswordInput
        v-model="form.confirmPassword"
        label="确认密码"
        placeholder="请再次输入新密码"
        :error="errors.confirmPassword"
        required
      />

      <Alert v-if="resetError" variant="danger" :message="resetError" />

      <Button type="submit" variant="primary" block :loading="loading">
        重置密码
      </Button>
    </form>

    <!-- 步骤4: 重置成功 -->
    <div v-else-if="step === 4" class="space-y-5">
      <div class="text-center py-6">
        <div class="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-success-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h3 class="text-lg font-semibold text-secondary-900 mb-2">密码重置成功</h3>
        <p class="text-secondary-500">
          您的密码已成功重置，请使用新密码登录
        </p>
      </div>

      <Button variant="primary" block @click="router.push('/login')">
        返回登录
      </Button>
    </div>

    <div v-if="step === 1" class="text-center">
      <router-link to="/login" class="text-sm text-secondary-500 hover:text-secondary-700">
        <span class="flex items-center justify-center gap-1">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          返回登录
        </span>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import Input from '@/components/ui/Input.vue'
import PasswordInput from '@/components/ui/PasswordInput.vue'
import Button from '@/components/ui/Button.vue'
import Alert from '@/components/ui/Alert.vue'

const router = useRouter()
const route = useRoute()

const step = ref(1) // 1: 输入邮箱, 2: 邮件已发送, 3: 设置新密码, 4: 重置成功
const loading = ref(false)
const resending = ref(false)
const sendError = ref('')
const resetError = ref('')

const form = reactive({
  email: '',
  newPassword: '',
  confirmPassword: ''
})

const errors = reactive({
  email: '',
  newPassword: '',
  confirmPassword: ''
})

onMounted(() => {
  // 检查是否有 token 参数，如果有则进入重置密码步骤
  if (route.query.token && route.query.email) {
    form.email = route.query.email
    step.value = 3
  }
})

const validateEmail = () => {
  errors.email = ''
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  
  if (!form.email) {
    errors.email = '请输入邮箱地址'
    return false
  }
  if (!emailRegex.test(form.email)) {
    errors.email = '请输入有效的邮箱地址'
    return false
  }
  return true
}

const validatePassword = () => {
  errors.newPassword = ''
  errors.confirmPassword = ''
  let isValid = true

  if (!form.newPassword) {
    errors.newPassword = '请输入新密码'
    isValid = false
  } else if (form.newPassword.length < 8) {
    errors.newPassword = '密码长度至少为8位'
    isValid = false
  } else if (!/[a-zA-Z]/.test(form.newPassword) || !/\d/.test(form.newPassword)) {
    errors.newPassword = '密码必须包含字母和数字'
    isValid = false
  }

  if (!form.confirmPassword) {
    errors.confirmPassword = '请确认密码'
    isValid = false
  } else if (form.newPassword !== form.confirmPassword) {
    errors.confirmPassword = '两次输入的密码不一致'
    isValid = false
  }

  return isValid
}

const handleSendEmail = async () => {
  if (!validateEmail()) return

  loading.value = true
  sendError.value = ''

  try {
    // TODO: 调用 API 发送重置邮件
    await new Promise(resolve => setTimeout(resolve, 1500))
    step.value = 2
  } catch (error) {
    sendError.value = error.message || '发送失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

const handleResend = async () => {
  resending.value = true
  sendError.value = ''

  try {
    // TODO: 调用 API 重新发送邮件
    await new Promise(resolve => setTimeout(resolve, 1500))
  } catch (error) {
    sendError.value = error.message || '发送失败，请稍后重试'
  } finally {
    resending.value = false
  }
}

const handleResetPassword = async () => {
  if (!validatePassword()) return

  loading.value = true
  resetError.value = ''

  try {
    // TODO: 调用 API 重置密码
    await new Promise(resolve => setTimeout(resolve, 1500))
    step.value = 4
  } catch (error) {
    resetError.value = error.message || '重置失败，请稍后重试'
  } finally {
    loading.value = false
  }
}
</script>
