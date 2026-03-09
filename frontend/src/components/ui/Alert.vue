<template>
  <div :class="alertClasses">
    <div class="flex-shrink-0">
      <component :is="iconComponent" class="w-5 h-5" />
    </div>
    <div class="flex-1">
      <h4 v-if="title" class="text-sm font-semibold mb-1">{{ title }}</h4>
      <p class="text-sm">{{ message }}</p>
      <slot />
    </div>
    <button
      v-if="closable"
      @click="handleClose"
      class="flex-shrink-0 p-1 rounded-lg hover:bg-black/10 transition-colors"
    >
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>
</template>

<script setup>
import { computed, h } from 'vue'

const props = defineProps({
  variant: {
    type: String,
    default: 'info',
    validator: (v) => ['info', 'success', 'warning', 'danger'].includes(v)
  },
  title: {
    type: String,
    default: ''
  },
  message: {
    type: String,
    default: ''
  },
  closable: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close'])

const alertClasses = computed(() => {
  const classes = ['p-4 rounded-lg flex items-start gap-3']

  const variantClasses = {
    info: 'bg-primary-50 text-primary-800 border border-primary-200',
    success: 'bg-success-50 text-success-800 border border-success-200',
    warning: 'bg-warning-50 text-warning-800 border border-warning-200',
    danger: 'bg-danger-50 text-danger-800 border border-danger-200'
  }
  classes.push(variantClasses[props.variant])

  return classes.join(' ')
})

const iconComponent = computed(() => {
  const icons = {
    info: {
      render() {
        return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    },
    success: {
      render() {
        return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    },
    warning: {
      render() {
        return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' })
        ])
      }
    },
    danger: {
      render() {
        return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
          h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z' })
        ])
      }
    }
  }
  return icons[props.variant]
})

const handleClose = () => {
  emit('close')
}
</script>
