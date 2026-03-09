<template>
  <div class="w-full">
    <label v-if="label" class="form-label">
      {{ label }}
    </label>
    <div class="relative">
      <input
        type="date"
        :value="modelValue"
        :disabled="disabled"
        :min="min"
        :max="max"
        class="form-input"
        @input="handleInput"
      />
    </div>
    <p v-if="error" class="form-error">{{ error }}</p>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: {
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
  },
  min: {
    type: String,
    default: ''
  },
  max: {
    type: String,
    default: ''
  },
  error: {
    type: String,
    default: ''
  }
})

const emit = defineEmits(['update:modelValue'])

const handleInput = (event) => {
  emit('update:modelValue', event.target.value)
}
</script>
