<template>
  <div class="w-full">
    <label v-if="label" :for="id" class="form-label">
      {{ label }}
      <span v-if="required" class="text-danger-500">*</span>
    </label>
    <div class="relative">
      <div v-if="$slots.prefix" class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <slot name="prefix" />
      </div>
      <input
        :id="id"
        :type="type"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        :class="inputClasses"
        @input="handleInput"
        @focus="handleFocus"
        @blur="handleBlur"
      />
      <div v-if="$slots.suffix" class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        <slot name="suffix" />
      </div>
    </div>
    <p v-if="error" class="form-error">{{ error }}</p>
    <p v-else-if="hint" class="form-hint">{{ hint }}</p>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  id: {
    type: String,
    default: () => `input-${Math.random().toString(36).slice(2, 11)}`
  },
  modelValue: {
    type: [String, Number],
    default: ''
  },
  type: {
    type: String,
    default: 'text'
  },
  label: {
    type: String,
    default: ''
  },
  placeholder: {
    type: String,
    default: ''
  },
  disabled: {
    type: Boolean,
    default: false
  },
  readonly: {
    type: Boolean,
    default: false
  },
  required: {
    type: Boolean,
    default: false
  },
  error: {
    type: String,
    default: ''
  },
  hint: {
    type: String,
    default: ''
  },
  size: {
    type: String,
    default: 'md',
    validator: (v) => ['sm', 'md', 'lg'].includes(v)
  }
})

const emit = defineEmits(['update:modelValue', 'focus', 'blur'])

const isFocused = ref(false)

const inputClasses = computed(() => {
  const classes = ['form-input']

  if (props.error) {
    classes.push('border-danger-500 focus:ring-danger-500')
  }

  if (props.disabled) {
    classes.push('bg-secondary-100 cursor-not-allowed')
  }

  // Size classes
  const sizeClasses = {
    sm: 'py-1.5 text-sm',
    md: 'py-2.5',
    lg: 'py-3 text-lg'
  }
  classes.push(sizeClasses[props.size])

  return classes.join(' ')
})

const handleInput = (event) => {
  emit('update:modelValue', event.target.value)
}

const handleFocus = (event) => {
  isFocused.value = true
  emit('focus', event)
}

const handleBlur = (event) => {
  isFocused.value = false
  emit('blur', event)
}
</script>
