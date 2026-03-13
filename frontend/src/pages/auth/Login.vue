<template>
  <div class="space-y-8">
    <!-- 头部 -->
    <div class="text-center">
      <h2 class="text-3xl font-bold text-secondary-900 mb-2">欢迎回来</h2>
      <p class="text-secondary-500">登录 PythonDL 智能分析平台</p>
    </div>

    <!-- 登录表单 -->
    <form @submit.prevent="handleLogin" class="space-y-5">
      <!-- 用户名输入 -->
      <div class="form-group">
        <label class="form-label form-label-required">用户名</label>
        <div class="relative">
          <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <svg class="h-5 w-5 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <input
            v-model="form.username"
            type="text"
            class="form-input pl-11"
            placeholder="请输入用户名"
            :class="{ 'form-input-error': errors.username }"
            required
          />
        </div>
        <p v-if="errors.username" class="form-error">{{ errors.username }}</p>
      </div>

      <!-- 密码输入 -->
      <div class="form-group">
        <label class="form-label form-label-required">密码</label>
        <div class="relative">
          <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <svg class="h-5 w-5 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <input
            v-model="form.password"
            :type="showPassword ? 'text' : 'password'"
            class="form-input pl-11 pr-11"
            placeholder="请输入密码"
            :class="{ 'form-input-error': errors.password }"
            required
          />
          <button
            type="button"
            @click="showPassword = !showPassword"
            class="absolute inset-y-0 right-0 pr-4 flex items-center text-secondary-400 hover:text-secondary-600"
          >
            <svg v-if="!showPassword" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <svg v-else class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
            </svg>
          </button>
        </div>
        <p v-if="errors.password" class="form-error">{{ errors.password }}</p>
      </div>

      <!-- 记住我 & 忘记密码 -->
      <div class="flex items-center justify-between">
        <label class="flex items-center gap-2 cursor-pointer group">
          <input
            v-model="form.remember"
            type="checkbox"
            class="w-4 h-4 text-primary-600 border-secondary-300 rounded focus:ring-primary-500 cursor-pointer"
          />
          <span class="text-sm text-secondary-600 group-hover:text-secondary-800">记住我</span>
        </label>
        <router-link
          to="/forgot-password"
          class="text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors"
        >
          忘记密码？
        </router-link>
      </div>

      <!-- 错误提示 -->
      <Transition
        enter-active-class="transition ease-out duration-200"
        enter-from-class="opacity-0 -translate-y-2"
        enter-to-class="opacity-100 translate-y-0"
        leave-active-class="transition ease-in duration-150"
        leave-from-class="opacity-100 translate-y-0"
        leave-to-class="opacity-0 -translate-y-2"
      >
        <div v-if="loginError" class="alert alert-danger">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <p class="alert-title">登录失败</p>
            <p class="alert-description">{{ loginError }}</p>
          </div>
        </div>
      </Transition>

      <!-- 登录按钮 -->
      <button
        type="submit"
        :disabled="loading"
        class="btn btn-primary btn-block btn-lg"
      >
        <svg v-if="loading" class="animate-spin -ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span v-if="loading">登录中...</span>
        <span v-else>登录</span>
      </button>
    </form>

    <!-- 分割线 -->
    <div class="relative">
      <div class="absolute inset-0 flex items-center">
        <div class="w-full border-t border-secondary-200"></div>
      </div>
      <div class="relative flex justify-center text-sm">
        <span class="px-4 bg-white text-secondary-500 font-medium">或使用以下方式登录</span>
      </div>
    </div>

    <!-- OAuth 登录 -->
    <div class="grid grid-cols-2 gap-3">
      <button
        @click="handleOAuthLogin('github')"
        class="btn btn-outline hover:bg-secondary-50"
      >
        <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
          <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd" />
        </svg>
        GitHub
      </button>
      <button
        @click="handleOAuthLogin('google')"
        class="btn btn-outline hover:bg-secondary-50"
      >
        <svg class="w-5 h-5 mr-2" viewBox="0 0 24 24">
          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
        </svg>
        Google
      </button>
    </div>

    <!-- 注册链接 -->
    <p class="text-center text-sm text-secondary-600">
      还没有账户？
      <router-link
        to="/register"
        class="font-semibold text-primary-600 hover:text-primary-700 transition-colors"
      >
        立即注册
      </router-link>
    </p>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const loading = ref(false)
const loginError = ref('')
const showPassword = ref(false)

const form = reactive({
  username: '',
  password: '',
  remember: false
})

const errors = reactive({
  username: '',
  password: ''
})

const validateForm = () => {
  let isValid = true
  errors.username = ''
  errors.password = ''

  if (!form.username) {
    errors.username = '请输入用户名'
    isValid = false
  }

  if (!form.password) {
    errors.password = '请输入密码'
    isValid = false
  } else if (form.password.length < 6) {
    errors.password = '密码长度至少为 6 位'
    isValid = false
  }

  return isValid
}

const handleLogin = async () => {
  if (!validateForm()) return

  loading.value = true
  loginError.value = ''

  try {
    await authStore.login({
      username: form.username,
      password: form.password,
      remember: form.remember
    })
    router.push('/dashboard')
  } catch (error) {
    loginError.value = error.message || '登录失败，请检查用户名和密码'
  } finally {
    loading.value = false
  }
}

const handleOAuthLogin = (provider) => {
  console.log('OAuth login:', provider)
  // TODO: Implement OAuth login
}
</script>
