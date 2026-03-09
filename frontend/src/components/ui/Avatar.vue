<template>
  <div class="flex items-center gap-2">
    <div
      :class="[
        'w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold',
        sizeClasses[size],
        colorClasses[color]
      ]"
    >
      <img v-if="src" :src="src" :alt="alt" class="w-full h-full rounded-full object-cover" />
      <span v-else>{{ initials }}</span>
    </div>
    <div v-if="showInfo" class="flex flex-col">
      <span class="text-sm font-medium text-secondary-900">{{ name }}</span>
      <span class="text-xs text-secondary-500">{{ subtitle }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  src: {
    type: String,
    default: ''
  },
  alt: {
    type: String,
    default: ''
  },
  name: {
    type: String,
    default: ''
  },
  subtitle: {
    type: String,
    default: ''
  },
  size: {
    type: String,
    default: 'md',
    validator: (v) => ['sm', 'md', 'lg', 'xl'].includes(v)
  },
  color: {
    type: String,
    default: 'primary',
    validator: (v) => ['primary', 'secondary', 'success', 'warning', 'danger'].includes(v)
  },
  showInfo: {
    type: Boolean,
    default: false
  }
})

const initials = computed(() => {
  if (props.name) {
    return props.name.slice(0, 2).toUpperCase()
  }
  return 'U'
})

const sizeClasses = {
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
  xl: 'w-16 h-16 text-lg'
}

const colorClasses = {
  primary: 'bg-primary-100 text-primary-600',
  secondary: 'bg-secondary-100 text-secondary-600',
  success: 'bg-success-100 text-success-600',
  warning: 'bg-warning-100 text-warning-600',
  danger: 'bg-danger-100 text-danger-600'
}
</script>
