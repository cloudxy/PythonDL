<template>
  <Teleport to="body">
    <Transition
      enter-active-class="transition ease-out duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition ease-in duration-150"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div v-if="modelValue" class="modal-overlay" @click="handleOverlayClick"></div>
    </Transition>
    
    <Transition
      enter-active-class="transition ease-out duration-200"
      enter-from-class="opacity-0 scale-95"
      enter-to-class="opacity-100 scale-100"
      leave-active-class="transition ease-in duration-150"
      leave-from-class="opacity-100 scale-100"
      leave-to-class="opacity-0 scale-95"
    >
      <div v-if="modelValue" class="modal-container" @click.self="handleOverlayClick">
        <div :class="modalClasses">
          <!-- Header -->
          <div class="modal-header flex items-center justify-between">
            <div>
              <h3 class="text-lg font-semibold text-secondary-900">{{ title }}</h3>
              <p v-if="subtitle" class="text-sm text-secondary-500 mt-0.5">{{ subtitle }}</p>
            </div>
            <button
              v-if="closable"
              @click="close"
              class="p-1 text-secondary-400 hover:text-secondary-600 rounded-lg hover:bg-secondary-100 transition-colors"
            >
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Body -->
          <div class="modal-body" :style="{ maxHeight: props.maxHeight }">
            <slot />
          </div>

          <!-- Footer -->
          <div v-if="$slots.footer || showDefaultFooter" class="modal-footer">
            <slot name="footer">
              <Button variant="ghost" @click="close">{{ cancelText }}</Button>
              <Button :variant="confirmVariant" :loading="loading" @click="handleConfirm">
                {{ confirmText }}
              </Button>
            </slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { computed, watch } from 'vue'
import Button from './Button.vue'

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  title: {
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
    validator: (v) => ['sm', 'md', 'lg', 'xl', 'full'].includes(v)
  },
  closable: {
    type: Boolean,
    default: true
  },
  closeOnOverlay: {
    type: Boolean,
    default: true
  },
  showDefaultFooter: {
    type: Boolean,
    default: false
  },
  confirmText: {
    type: String,
    default: '确定'
  },
  cancelText: {
    type: String,
    default: '取消'
  },
  confirmVariant: {
    type: String,
    default: 'primary'
  },
  loading: {
    type: Boolean,
    default: false
  },
  // 内容区域最大高度，支持滚动
  maxHeight: {
    type: String,
    default: '70vh'  // 默认最大高度为视口的 70%
  }
})

const emit = defineEmits(['update:modelValue', 'confirm', 'close'])

const modalClasses = computed(() => {
  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-[90vw]'
  }
  return `modal-content ${sizeClasses[props.size]}`
})

const close = () => {
  emit('update:modelValue', false)
  emit('close')
}

const handleOverlayClick = () => {
  if (props.closeOnOverlay) {
    close()
  }
}

const handleConfirm = () => {
  emit('confirm')
}

// Prevent body scroll when modal is open
watch(() => props.modelValue, (value) => {
  if (value) {
    document.body.style.overflow = 'hidden'
  } else {
    document.body.style.overflow = ''
  }
})
</script>
