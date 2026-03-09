<template>
  <div class="card">
    <div class="p-4 border-b border-secondary-100">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
    </div>
    <div class="p-4">
      <div class="flex items-center gap-4">
        <div
          v-for="(step, index) in steps"
          :key="index"
          class="flex items-center"
        >
          <div class="flex flex-col items-center">
            <div
              :class="[
                'w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-colors',
                index < currentStep
                  ? 'bg-success-500 text-white'
                  : index === currentStep
                    ? 'bg-primary-500 text-white'
                    : 'bg-secondary-200 text-secondary-500'
              ]"
            >
              <svg v-if="index < currentStep" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              <span v-else>{{ index + 1 }}</span>
            </div>
            <span
              :class="[
                'text-xs mt-2 transition-colors',
                index <= currentStep ? 'text-secondary-900 font-medium' : 'text-secondary-500'
              ]"
            >
              {{ step }}
            </span>
          </div>
          <div
            v-if="index < steps.length - 1"
            :class="[
              'w-12 h-0.5 mx-2 transition-colors',
              index < currentStep ? 'bg-success-500' : 'bg-secondary-200'
            ]"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: '进度'
  },
  steps: {
    type: Array,
    required: true
  },
  currentStep: {
    type: Number,
    default: 0
  }
})
</script>
