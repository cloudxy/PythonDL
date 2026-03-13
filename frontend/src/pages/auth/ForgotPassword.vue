<template>
  <div class="space-y-8">
    <!-- 头部 -->
    <div class="text-center">
      <div class="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-2xl mb-4">
        <svg class="w-8 h-8 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
        </svg>
      </div>
      <h2 class="text-3xl font-bold text-secondary-900 mb-2">忘记密码</h2>
      <p class="text-secondary-500">别担心，我们会帮您重置密码</p>
    </div>

    <!-- 步骤指示器 -->
    <div class="flex items-center justify-center">
      <div class="flex items-center gap-2">
        <div
          :class="[
            'flex items-center justify-center w-8 h-8 rounded-full font-semibold text-sm transition-all duration-300',
            currentStep >= 1 ? 'bg-primary-600 text-white' : 'bg-secondary-200 text-secondary-600'
          ]"
        >
          1
        </div>
        <div :class="['w-12 h-1 rounded-full transition-all duration-300', currentStep >= 2 ? 'bg-primary-600' : 'bg-secondary-200']"></div>
        <div
          :class="[
            'flex items-center justify-center w-8 h-8 rounded-full font-semibold text-sm transition-all duration-300',
            currentStep >= 2 ? 'bg-primary-600 text-white' : 'bg-secondary-200 text-secondary-600'
          ]"
        >
          2
        </div>
      </div>
    </div>

    <!-- 步骤 1: 输入邮箱 -->
    <Transition
      v-if="currentStep === 1"
      enter-active-class="transition ease-out duration-300"
      enter-from-class="opacity-0 translate-x-10"
      enter-to-class="opacity-100 translate-x-0"
      leave-active-class="transition ease-in duration-200"
      leave-from-class="opacity-100 translate-x-0"
      leave-to-class="opacity-0 -translate-x-10"
    >
      <form @submit.prevent="handleSendResetEmail" class="space-y-5">
        <div class="alert alert-info">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p class="text-sm">输入您注册时使用的邮箱，我们将发送密码重置链接</p>
        </div>

        <div class="form-group">
          <label class="form-label form-label-required">电子邮箱</label>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <svg class="h-5 w-5 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <input
              v-model="form.email"
              type="email"
              class="form-input pl-11"
              placeholder="example@email.com"
              :class="{ 'form-input-error': errors.email }"
              required
            />
          </div>
          <p v-if="errors.email" class="form-error">{{ errors.email }}</p>
        </div>

        <Transition
          enter-active-class="transition ease-out duration-200"
          enter-from-class="opacity-0 -translate-y-2"
          enter-to-class="opacity-100 translate-y-0"
          leave-active-class="transition ease-in duration-150"
          leave-from-class="opacity-100 translate-y-0"
          leave-to-class="opacity-0 -translate-y-2"
        >
          <div v-if="sendError" class="alert alert-danger">
            <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p class="alert-title">发送失败</p>
              <p class="alert-description">{{ sendError }}</p>
            </div>
          </div>
        </Transition>

        <button
          type="submit"
          :disabled="loading"
          class="btn btn-primary btn-block btn-lg"
        >
          <svg v-if="loading" class="animate-spin -ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span v-if="loading">发送中...</span>
          <span v-else>发送重置链接</span>
        </button>
      </form>
    </Transition>

    <!-- 步骤 2: 检查邮箱 -->
    <Transition
      v-if="currentStep === 2"
      enter-active-class="transition ease-out duration-300"
      enter-from-class="opacity-0 translate-x-10"
      enter-to-class="opacity-100 translate-x-0"
      leave-active-class="transition ease-in duration-200"
      leave-from-class="opacity-100 translate-x-0"
      leave-to-class="opacity-0 -translate-x-10"
    >
      <div class="text-center py-8">
        <div class="inline-flex items-center justify-center w-20 h-20 bg-success-100 rounded-full mb-6">
          <svg class="w-10 h-10 text-success-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        
        <h3 class="text-xl font-semibold text-secondary-900 mb-2">邮件已发送</h3>
        <p class="text-secondary-500 mb-6">
          我们已向 <span class="font-medium text-secondary-900">{{ form.email }}</span> 发送了密码重置链接
        </p>

        <div class="alert alert-info mb-6">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p class="text-sm">请检查您的收件箱，如果没有收到邮件，请检查垃圾邮件文件夹</p>
        </div>

        <div class="space-y-3">
          <button
            @click="handleResend"
            :disabled="resendLoading || resendCountdown > 0"
            class="btn btn-secondary btn-block"
          >
            <span v-if="resendLoading">发送中...</span>
            <span v-else-if="resendCountdown > 0">{{ resendCountdown }}秒后重发</span>
            <span v-else>重新发送邮件</span>
          </button>

          <button
            @click="currentStep = 1"
            :disabled="resendLoading"
            class="btn btn-ghost btn-block"
          >
            更换邮箱
          </button>
        </div>
      </div>
    </Transition>

    <!-- 返回登录 -->
    <div class="text-center pt-4 border-t border-secondary-100">
      <router-link
        to="/login"
        class="inline-flex items-center gap-2 text-sm font-medium text-secondary-600 hover:text-secondary-900 transition-colors"
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        返回登录
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { authApi } from '@/api/auth'

const currentStep = ref(1)
const loading = ref(false)
const sendError = ref('')
const resendLoading = ref(false)
const resendCountdown = ref(0)

const form = reactive({
  email: ''
})

const errors = reactive({
  email: ''
})

const validateEmail = () => {
  errors.email = ''
  
  if (!form.email) {
    errors.email = '请输入邮箱地址'
    return false
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(form.email)) {
    errors.email = '请输入有效的邮箱地址'
    return false
  }
  
  return true
}

const handleSendResetEmail = async () => {
  if (!validateEmail()) return

  loading.value = true
  sendError.value = ''

  try {
    await authApi.forgotPassword({ email: form.email })
    currentStep.value = 2
    startCountdown()
  } catch (error) {
    sendError.value = error.message || '发送失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

const startCountdown = () => {
  resendCountdown.value = 60
  const timer = setInterval(() => {
    resendCountdown.value--
    if (resendCountdown.value <= 0) {
      clearInterval(timer)
    }
  }, 1000)
}

const handleResend = async () => {
  if (resendCountdown.value > 0) return

  resendLoading.value = true

  try {
    await authApi.forgotPassword({ email: form.email })
    startCountdown()
  } catch (error) {
    sendError.value = error.message || '发送失败，请稍后重试'
  } finally {
    resendLoading.value = false
  }
}
</script>
