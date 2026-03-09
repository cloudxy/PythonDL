<template>
  <div class="min-h-screen flex flex-col bg-gradient-to-br from-primary-600 via-primary-700 to-secondary-900">
    <!-- 顶部装饰 -->
    <div class="absolute inset-0 overflow-hidden pointer-events-none">
      <div class="absolute -top-40 -right-40 w-80 h-80 bg-primary-400/20 rounded-full blur-3xl"></div>
      <div class="absolute -bottom-40 -left-40 w-80 h-80 bg-secondary-400/20 rounded-full blur-3xl"></div>
      <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl"></div>
    </div>

    <!-- 主内容区域 -->
    <div class="flex-1 flex items-center justify-center p-4 relative z-10">
      <div class="w-full max-w-md">
        <!-- Logo 和标题 -->
        <div class="text-center mb-8">
          <div class="inline-flex items-center justify-center w-16 h-16 bg-white/10 backdrop-blur-sm rounded-2xl mb-4">
            <svg class="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h1 class="text-3xl font-bold text-white mb-2">PythonDL</h1>
          <p class="text-primary-200 text-sm">智能分析平台</p>
        </div>

        <!-- 内容卡片 -->
        <div class="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8">
          <router-view />
        </div>

        <!-- 底部链接 -->
        <div class="mt-6 text-center">
          <p class="text-primary-200 text-sm">
            &copy; {{ currentYear }} PythonDL. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const currentYear = computed(() => new Date().getFullYear())
</script>
