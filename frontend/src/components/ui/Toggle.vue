<template>
  <div class="flex items-center gap-2">
    <div
      :class="[
        'w-11 h-6 rounded-full relative cursor-pointer transition-colors',
        modelValue ? 'bg-primary-600' : 'bg-secondary-300',
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      ]"
      @click="handleToggle"
    >
      <div
        :class="[
          'absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform',
          modelValue ? 'translate-x-5.5 left-0.5' : 'left-0.5'
        ]"
      ></div>
    </div>
    <span v-if="label" class="text-sm text-secondary-700">{{ label }}</span>
  </div>
</template>

<script setup>
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

const handleToggle = () => {
  if (!props.disabled) {
    emit('update:modelValue', !props.modelValue)
  }
}
</script>

<style scoped>
.translate-x-5\.5 {
  transform: translateX(1.375rem);
}
</style>
