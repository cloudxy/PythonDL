<template>
  <div class="w-full">
    <label :class="labelClasses">
      <span class="relative">
        <input
          type="radio"
          :name="name"
          :value="value"
          :checked="modelValue === value"
          :disabled="disabled"
          class="sr-only"
          @change="handleChange"
        />
        <span :class="radioClasses">
          <span
            v-if="modelValue === value"
            class="w-2.5 h-2.5 rounded-full bg-white"
          ></span>
        </span>
      </span>
      <span v-if="label" class="text-sm text-secondary-700">{{ label }}</span>
    </label>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: {
    type: [String, Number, Boolean],
    default: ''
  },
  value: {
    type: [String, Number, Boolean],
    required: true
  },
  name: {
    type: String,
    default: ''
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

const radioClasses = computed(() => {
  const classes = [
    'w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors'
  ]
  if (props.modelValue === props.value) {
    classes.push('bg-primary-600 border-primary-600')
  } else {
    classes.push('border-secondary-300 bg-white')
  }
  return classes.join(' ')
})

const handleChange = () => {
  if (!props.disabled) {
    emit('update:modelValue', props.value)
  }
}
</script>
