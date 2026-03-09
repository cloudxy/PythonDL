<template>
  <div class="space-y-4">
    <div v-for="(item, index) in items" :key="index" class="flex gap-3">
      <div class="flex-shrink-0">
        <div
          :class="[
            'w-8 h-8 rounded-full flex items-center justify-center',
            dotColorClass(item.type)
          ]"
        >
          <component :is="iconComponent(item.type)" class="w-4 h-4 text-white" />
        </div>
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm text-secondary-900">{{ item.content }}</p>
        <p class="text-xs text-secondary-500 mt-1">{{ item.time }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { h } from 'vue'

defineProps({
  items: {
    type: Array,
    default: () => []
  }
})

const dotColorClass = (type) => {
  const classes = {
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    info: 'bg-primary-500'
  }
  return classes[type] || 'bg-secondary-500'
}

const iconComponent = (type) => {
  const icons = {
    success: {
      render() {
        return h('svg', { class: 'w-4 h-4', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M5 13l4 4L19 7' })
        ])
      }
    },
    warning: {
      render() {
        return h('svg', { class: 'w-4 h-4', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M12 9v2m0 4h.01' })
        ])
      }
    },
    danger: {
      render() {
        return h('svg', { class: 'w-4 h-4', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M6 18L18 6M6 6l12 12' })
        ])
      }
    },
    info: {
      render() {
        return h('svg', { class: 'w-4 h-4', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M13 16h-1v-4h-1m1-4h.01' })
        ])
      }
    }
  }
  return icons[type] || icons.info
}
</script>
