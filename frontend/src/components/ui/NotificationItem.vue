<template>
  <div class="flex items-center gap-3 p-4 bg-secondary-50 rounded-lg">
    <div class="flex-shrink-0">
      <component :is="iconComponent" class="w-6 h-6" :class="iconClass" />
    </div>
    <div class="flex-1 min-w-0">
      <p class="text-sm font-medium text-secondary-900">{{ title }}</p>
      <p class="text-sm text-secondary-500 truncate">{{ description }}</p>
    </div>
    <Button v-if="actionText" variant="ghost" size="sm" @click="$emit('action')">
      {{ actionText }}
    </Button>
  </div>
</template>

<script setup>
import { computed, h } from 'vue'
import Button from './Button.vue'

const props = defineProps({
  icon: {
    type: String,
    default: 'info',
    validator: (v) => ['info', 'success', 'warning', 'danger'].includes(v)
  },
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  },
  actionText: {
    type: String,
    default: ''
  }
})

defineEmits(['action'])

const iconComponent = computed(() => {
  const icons = {
    info: {
      render() {
        return h('svg', { class: 'w-6 h-6', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    },
    success: {
      render() {
        return h('svg', { class: 'w-6 h-6', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    },
    warning: {
      render() {
        return h('svg', { class: 'w-6 h-6', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' })
        ])
      }
    },
    danger: {
      render() {
        return h('svg', { class: 'w-6 h-6', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    }
  }
  return icons[props.icon]
})

const iconClass = computed(() => {
  const classes = {
    info: 'text-primary-500',
    success: 'text-success-500',
    warning: 'text-warning-500',
    danger: 'text-danger-500'
  }
  return classes[props.icon]
})
</script>
