<template>
  <div class="w-full">
    <label :class="labelClasses">
      <input
        type="checkbox"
        :checked="modelValue"
        :disabled="disabled"
        class="sr-only"
        @change="handleChange"
      />
      <span :class="checkboxClasses">
        <svg
          v-if="modelValue"
          class="w-3 h-3 text-white"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="3"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      </span>
      <span v-if="label" class="text-sm text-secondary-700">{{ label }}</span>
    </label>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  label: {
    type: String,
    default: ''
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue'])

const labelClasses = computed(() => {
  const classes = ['inline-flex items-center gap-2 cursor-pointer']
  if (props.disabled) {
    classes.push('opacity-50 cursor-not-allowed')
  }
  return classes.join(' ')
})

const checkboxClasses = computed(() => {
  const classes = [
    'w-5 h-5 rounded border-2 flex items-center justify-center transition-colors'
  ]
  if (props.modelValue) {
    classes.push('bg-primary-600 border-primary-600')
  } else {
    classes.push('border-secondary-300 bg-white')
  }
  return classes.join(' ')
})

const handleChange = () => {
  if (!props.disabled) {
    emit('update:modelValue', !props.modelValue)
  }
}
</script>
