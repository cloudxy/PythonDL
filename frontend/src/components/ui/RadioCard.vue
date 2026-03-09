<template>
  <div class="w-full">
    <label v-if="label" class="form-label">
      {{ label }}
    </label>
    <div class="flex items-center gap-4">
      <div
        v-for="option in options"
        :key="option.value"
        :class="[
          'flex-1 p-4 border-2 rounded-lg cursor-pointer transition-all',
          modelValue === option.value
            ? 'border-primary-500 bg-primary-50'
            : 'border-secondary-200 hover:border-secondary-300'
        ]"
        @click="handleSelect(option.value)"
      >
        <div class="flex items-center gap-3">
          <div
            :class="[
              'w-5 h-5 rounded-full border-2 flex items-center justify-center',
              modelValue === option.value
                ? 'border-primary-500 bg-primary-500'
                : 'border-secondary-300'
            ]"
          >
            <div
              v-if="modelValue === option.value"
              class="w-2 h-2 rounded-full bg-white"
            ></div>
          </div>
          <div>
            <p class="font-medium text-secondary-900">{{ option.label }}</p>
            <p v-if="option.description" class="text-sm text-secondary-500 mt-0.5">
              {{ option.description }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: {
    type: [String, Number],
    default: ''
  },
  label: {
    type: String,
    default: ''
  },
  options: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['update:modelValue'])

const handleSelect = (value) => {
  emit('update:modelValue', value)
}
</script>
