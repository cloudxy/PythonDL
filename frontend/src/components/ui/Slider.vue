<template>
  <div class="w-full">
    <label v-if="label" class="form-label">
      {{ label }}
    </label>
    <div class="relative">
      <input
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        :disabled="disabled"
        class="slider-input"
        @input="handleInput"
      />
      <div class="flex justify-between mt-2">
        <span class="text-sm text-secondary-500">{{ min }}</span>
        <span class="text-sm font-medium text-primary-600">{{ modelValue }}</span>
        <span class="text-sm text-secondary-500">{{ max }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: {
    type: Number,
    default: 0
  },
  label: {
    type: String,
    default: ''
  },
  min: {
    type: Number,
    default: 0
  },
  max: {
    type: Number,
    default: 100
  },
  step: {
    type: Number,
    default: 1
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue'])

const handleInput = (event) => {
  emit('update:modelValue', Number(event.target.value))
}
</script>

<style scoped>
.slider-input {
  @apply w-full h-2 bg-secondary-200 rounded-lg appearance-none cursor-pointer;
}

.slider-input::-webkit-slider-thumb {
  @apply appearance-none w-5 h-5 bg-primary-600 rounded-full cursor-pointer shadow-md;
}

.slider-input::-moz-range-thumb {
  @apply w-5 h-5 bg-primary-600 rounded-full cursor-pointer shadow-md border-0;
}
</style>
